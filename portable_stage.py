"""Execute the project's generated stage recipes without cmd.exe or a shell.

The existing .bat recipes remain the canonical, hashed stage configuration.
Only their small generated grammar is supported, on both Windows and Linux.
Parsing finishes before any command runs; unknown syntax fails closed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

MARKER = 'rem portable-stage-v1'


def condition(line):
    for pattern, kind in (
        (r'if (not )?exist "([^"]+)" (.+)', 'exists'),
        (r'if (not )?defined (\w+) (.+)', 'defined'),
        (r'if (not )?"([^"]*)"=="([^"]*)" (.+)', 'equals'),
    ):
        match = re.fullmatch(pattern, line, re.I)
        if match:
            fields = match.groups()
            return (kind, bool(fields[0]), *fields[1:-1]), fields[-1]
    raise ValueError('Unsupported generated condition: ' + line)


def statement(line):
    if line.startswith('if '):
        test, body = condition(line)
        return ('if', test, [statement(body)], [])
    if line.startswith('echo '):
        return ('echo', line[5:])
    if line.startswith(':'):
        return ('label', line[1:])
    if line.startswith('goto :'):
        return ('goto', line[6:])
    if line.startswith('exit /b '):
        return ('exit', line[8:])
    match = re.fullmatch(r'set "(\w+)=(.*)"', line)
    if match:
        return ('set', *match.groups())
    match = re.fullmatch(r'for /f "usebackq delims=" %%I in \(`(.+)`\) do set "(\w+)=%%I"', line)
    if match:
        return ('capture', match[2], command_words(match[1]))
    for operation in ('mkdir', 'del'):
        match = re.fullmatch(operation + r' "([^"]+)"', line)
        if match:
            return (operation, match[1])
    if line.startswith('python ') or line.startswith('"%COLMAP_EXE%" '):
        return ('command', command_words(line))
    raise ValueError('Unsupported generated statement: ' + line)


def command_words(line):
    # Tokenize before variable expansion: spaces/quotes/metacharacters in a
    # captured path stay one argument and are never evaluated as shell code.
    words = shlex.split(line, posix=False)
    return [word[1:-1] if word.startswith('"') and word.endswith('"') else word for word in words]


def parse_recipe(text):
    if MARKER not in text.splitlines():
        raise ValueError('Portable execution requires a newly generated stage recipe')
    lines, partial = [], ''
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in ('@echo off', 'setlocal') or line.startswith('rem '):
            continue
        partial += line[:-1] + ' ' if line.endswith('^') else line
        if not line.endswith('^'):
            lines.append(partial.strip()); partial = ''
    if partial:
        raise ValueError('Incomplete command continuation')

    def block(index, nested=False):
        nodes = []
        while index < len(lines):
            line = lines[index]; index += 1
            if line in (')', ') else ('):
                if not nested:
                    raise ValueError('Unexpected block terminator')
                return nodes, index, line
            if line.startswith('if ') and line.endswith(' ('):
                test, body = condition(line)
                yes, index, end = block(index, True)
                no = []
                if end == ') else (':
                    no, index, end = block(index, True)
                if end != ')':
                    raise ValueError('Unclosed conditional block')
                nodes.append(('if', test, yes, no))
            else:
                nodes.append(statement(line))
        if nested:
            raise ValueError('Unclosed conditional block')
        return nodes, index, None

    nodes, _, _ = block(0)
    labels = [node[1] for node in nodes if node[0] == 'label']
    if len(labels) != len(set(labels)):
        raise ValueError('Duplicate stage label')
    def validate(nodes):
        for node in nodes:
            if node[0] == 'goto' and node[1] not in labels:
                raise ValueError('Unknown stage label: ' + node[1])
            if node[0] == 'if':
                validate(node[2]); validate(node[3])
    validate(nodes)
    return nodes


class Jump(Exception):
    pass


class ExitStage(Exception):
    pass


class Stage:
    def __init__(self, script):
        self.script = Path(script).resolve()
        self.nodes = parse_recipe(self.script.read_text(encoding='utf-8'))
        self.values = {key.upper(): value for key, value in os.environ.items()}
        self.code = 0

    def expand(self, text):
        def value(match):
            name = match[1].upper()
            if name == 'ERRORLEVEL':
                return str(self.code)
            if name == 'RANDOM':
                return uuid.uuid4().hex[:8]
            return self.values.get(name, '')
        return re.sub(r'%([A-Za-z_][A-Za-z_0-9]*)%', value, text)

    def path(self, value):
        return Path(self.expand(value).replace('\\', '/'))

    def test(self, spec):
        kind, negate, *args = spec
        if kind == 'exists':
            result = self.path(args[0]).exists()
        elif kind == 'defined':
            result = bool(self.values.get(args[0].upper()))
        else:
            result = self.expand(args[0]) == self.expand(args[1])
        return not result if negate else result

    def launch(self, words, capture=False):
        command = [self.expand(word).replace('\\', '/') for word in words]
        # An empty optional flag expands to zero arguments in the generated BAT.
        command = [word for word in command if word]
        if command[0] == 'python':
            command[0] = sys.executable
        kwargs = {'cwd': self.script.parent, 'env': os.environ.copy()}
        kwargs['env'].update({key: value for key, value in self.values.items() if key.startswith('FYP_')})
        # Captured sparse-model paths must round-trip on Windows too, where a
        # Python pipe otherwise defaults to the active ANSI code page.
        kwargs['env']['PYTHONIOENCODING'] = 'utf-8'
        if capture:
            kwargs['stdout'] = subprocess.PIPE
        from process_control import managed_process
        with managed_process(command, **kwargs) as process:
            output, _ = process.communicate()
            self.code = process.returncode
        return output.decode('utf-8', errors='replace').strip() if output is not None else ''

    def execute(self, node):
        kind, *args = node
        if kind == 'if':
            for child in args[1] if self.test(args[0]) else args[2]:
                self.execute(child)
        elif kind == 'echo':
            print(self.expand(args[0]), flush=True)
        elif kind == 'set':
            self.values[args[0].upper()] = self.expand(args[1])
        elif kind == 'command':
            self.launch(args[0])
        elif kind == 'capture':
            output = self.launch(args[1], capture=True)
            if self.code == 0 and output:
                self.values[args[0]] = output.splitlines()[-1]
        elif kind == 'mkdir':
            self.path(args[0]).mkdir(parents=True, exist_ok=True); self.code = 0
        elif kind == 'del':
            target = self.path(args[0]).resolve()
            if not target.is_relative_to(self.script.parent) or target.is_dir():
                raise ValueError('Refusing deletion outside the stage workspace: ' + str(target))
            target.unlink(missing_ok=True); self.code = 0
        elif kind == 'goto':
            raise Jump(args[0])
        elif kind == 'exit':
            raise ExitStage(int(self.expand(args[0])))

    def run(self):
        labels = {node[1]: i for i, node in enumerate(self.nodes) if node[0] == 'label'}
        index = 0
        try:
            while index < len(self.nodes):
                try:
                    self.execute(self.nodes[index]); index += 1
                except Jump as jump:
                    index = labels[jump.args[0]] + 1
        except ExitStage as result:
            return result.args[0]
        return self.code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', type=Path, required=True)
    args = parser.parse_args()
    try:
        return Stage(args.script).run()
    except (OSError, ValueError) as exc:
        print(f'[stage error] {exc}', flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
