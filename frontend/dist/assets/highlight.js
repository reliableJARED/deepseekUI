/* highlight.js — a small, dependency-free syntax highlighter.
 *
 * There are libraries for this: highlight.js (~190 languages, built-in
 * auto-detect), Prism, Shiki. This is deliberately none of them, for the same
 * reason markdown.js is hand-written — a "no build step, no CDN" frontend that
 * pulls in 120 KB of generated third-party code has quietly become neither. The
 * swap-in cost is low because everything goes through `highlightLines()`: vendoring
 * a real grammar engine later means reimplementing that one function.
 *
 * What it covers: the languages that actually turn up in a chat transcript —
 * Python, JS/TS, JSON, shell, SQL, Go, Rust, Java, C-family, Ruby, PHP, YAML,
 * TOML, Dockerfile, Markdown, HTML, CSS, diffs — plus a content-based guess for
 * blocks whose fence carried no language. Anything it does not know renders as
 * plain escaped text, which is exactly what it did before.
 *
 * The security model is markdown.js's, restated because it is the whole ballgame:
 * **every token's text is escaped before it is wrapped**, and the only HTML this
 * module can emit is `<span class="t-*">`. No input can become an element.
 *
 * Two invariants hold for every input, and they are what the harness pins down:
 *   1. Stripping the tags from `highlightLines(code, lang).join('\n')` and decoding
 *      the entities gives back `code` exactly — highlighting never adds, drops or
 *      reorders a character, including whitespace and line endings.
 *   2. Every `<span>` is closed before the end of its line, so a caller can zip
 *      line numbers against the result.
 */

const ESCAPE = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** Escape before wrapping, always. */
function esc(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ESCAPE[ch]);
}

const words = (text) => new Set(text.split(/\s+/).filter(Boolean));
const EMPTY = new Set();

const isDigit = (ch) => ch >= '0' && ch <= '9';
const isIdentStart = (ch) => /[A-Za-z_$]/.test(ch);
const isIdentPart = (ch) => /[\w$]/.test(ch);

/** Index of the next non-space character at or after `i`. */
function nextNonSpace(code, i) {
  let j = i;
  while (j < code.length && (code[j] === ' ' || code[j] === '\t')) j += 1;
  return j;
}

/**
 * Append a token, merging with the previous one when the class matches.
 *
 * Merging matters: a 40 KB file tokenised into one span per character would be
 * megabytes of HTML, and adjacent same-class tokens are indistinguishable to the
 * renderer anyway. A merged token may therefore contain newlines.
 */
function push(tokens, text, cls) {
  if (!text) return;
  const last = tokens[tokens.length - 1];
  if (last && last.cls === cls) last.text += text;
  else tokens.push({ text, cls });
}

/* ── language definitions ──────────────────────────────────────────────────── */

/* A "generic" language is described by data alone and scanned by `scanCode`:
 *
 *   line      sequences that start a comment running to end of line
 *   block     [open, close] for a comment that spans lines
 *   quote     characters that start a string
 *   template  a quote whose contents run across lines (` in JS, ` in shell)
 *   triple    a tripled quote opens a multi-line string (Python docstrings)
 *   prefix    letters that bind to a following string (Python's r"", rb"")
 *   keyword / type / literal / builtin    word classes
 *   decorator a character that introduces an annotation (@)
 *   variable  a character that introduces a variable ($)
 *   preproc   a line-leading directive (#include)
 *   props     an identifier or string followed by `:` or `=` names a property
 *   fold      match keywords case-insensitively (SQL)
 *
 * Anything omitted is simply not highlighted. */

const JS_KEYWORDS = 'as async await break case catch class const continue debugger default delete do else '
  + 'export extends finally for from function get if import in instanceof let new of return set static '
  + 'super switch this throw try typeof var void while with yield';
const JS_TYPES = 'Array Boolean Date Error Function JSON Map Math Number Object Promise Proxy Reflect RegExp '
  + 'Set String Symbol WeakMap WeakSet BigInt Intl ArrayBuffer DataView';
const JS_LITERALS = 'true false null undefined NaN Infinity globalThis arguments';
const JS_BUILTINS = 'console document window process require module exports fetch setTimeout setInterval '
  + 'clearTimeout clearInterval addEventListener removeEventListener querySelector querySelectorAll '
  + 'localStorage sessionStorage structuredClone alert prompt encodeURIComponent decodeURIComponent '
  + 'parseInt parseFloat isNaN requestAnimationFrame';

const SHELL_KEYWORDS = 'if then else elif fi for while until do done case esac function return in select time '
  + 'break continue exit local readonly export declare typeset unset shift source alias eval exec trap set '
  + 'function';
const SHELL_BUILTINS = 'echo printf cd pwd ls cp mv rm rmdir mkdir touch cat head tail grep egrep fgrep sed awk '
  + 'sort uniq wc cut tr find xargs tee chmod chown ln read test kill sleep curl wget git docker make python '
  + 'pip npm node clear du df ps top which basename dirname realpath';

const C_KEYWORDS = 'auto break case const continue default do else enum extern for goto if inline register '
  + 'restrict return sizeof static struct switch typedef union volatile while class namespace template public '
  + 'private protected virtual override final using new delete this nullptr operator explicit friend mutable '
  + 'try catch throw constexpr noexcept static_cast dynamic_cast reinterpret_cast const_cast typename '
  + 'wchar_t char16_t char32_t';
const C_TYPES = 'bool char double float int long short signed unsigned void size_t ssize_t int8_t int16_t '
  + 'int32_t int64_t uint8_t uint16_t uint32_t uint64_t string vector map set unordered_map FILE';

export const LANGS = {
  python: {
    label: 'python',
    line: ['#'], quote: '"\'', triple: true, prefix: 'rbufRBUF', decorator: '@',
    keyword: words('and as assert async await break case class continue def del elif else except finally for '
      + 'from global if import in is lambda match nonlocal not or pass raise return try while with yield'),
    type: words('bool bytes bytearray complex dict float frozenset int list object set str tuple type'),
    literal: words('True False None NotImplemented Ellipsis __name__ __main__ __file__'),
    builtin: words('abs all any ascii bin breakpoint callable chr classmethod compile delattr dir divmod '
      + 'enumerate eval exec filter format getattr globals hasattr hash help hex id input isinstance issubclass '
      + 'iter len locals map max min next oct open ord pow print property range repr reversed round setattr '
      + 'slice sorted staticmethod sum super vars zip'),
  },

  javascript: {
    label: 'javascript',
    line: ['//'], block: ['/*', '*/'], quote: '"\'', template: '`',
    keyword: words(JS_KEYWORDS), type: words(JS_TYPES),
    literal: words(JS_LITERALS), builtin: words(JS_BUILTINS),
  },

  typescript: {
    label: 'typescript',
    line: ['//'], block: ['/*', '*/'], quote: '"\'', template: '`',
    keyword: words(`${JS_KEYWORDS} interface type enum implements declare namespace abstract readonly `
      + 'keyof infer asserts satisfies override out is'),
    type: words(`${JS_TYPES} string number boolean any unknown never object symbol bigint`),
    literal: words(`${JS_LITERALS} never`), builtin: words(JS_BUILTINS),
  },

  json: {
    label: 'json',
    // Comment support is not standard JSON, but `.jsonc`, tsconfig.json and this
    // project's own hand-edited files all carry them and mis-highlighting is worse.
    line: ['//'], block: ['/*', '*/'], quote: '"',
    literal: words('true false null'), props: true,
  },

  shell: {
    label: 'bash',
    line: ['#'], quote: '"\'', template: '`', variable: '$',
    keyword: words(SHELL_KEYWORDS), literal: words('true false'),
    builtin: words(SHELL_BUILTINS),
  },

  powershell: {
    label: 'powershell',
    line: ['#'], block: ['<#', '#>'], quote: '"\'', variable: '$',
    keyword: words('-eq -ne -gt -lt -ge -le -and -or -not -like -match -contains -in -is function filter '
      + 'param begin process end if elseif else switch foreach while do until return break continue try catch '
      + 'finally throw class enum using module import export'),
    type: words('string int long double decimal bool datetime array hashtable pscustomobject switch object'),
    literal: words('$true $false $null'),
    builtin: words('Write-Host Write-Output Write-Error Get-ChildItem Get-Content Set-Content Select-Object '
      + 'Where-Object ForEach-Object Measure-Object Sort-Object Join-Path Test-Path New-Item Remove-Item Copy-Item'),
  },

  sql: {
    label: 'sql',
    line: ['--'], block: ['/*', '*/'], quote: '\'"', fold: true,
    keyword: words('select from where insert into values update delete create table alter drop add column '
      + 'index view trigger join left right inner outer full cross on group by order having limit offset as '
      + 'and or not null is in between like ilike distinct union all case when then else end primary key '
      + 'foreign references unique default check cascade constraint with returning set asc desc exists using '
      + 'cast begin commit rollback transaction explain analyze truncate grant revoke over partition'),
    type: words('int integer smallint bigint serial text varchar char boolean bool date timestamp timestamptz '
      + 'time numeric decimal real double float uuid json jsonb bytea array'),
    literal: words('true false null'),
    builtin: words('count sum avg min max coalesce nullif now date_trunc lower upper trim length substring '
      + 'concat round abs generate_series row_number'),
  },

  go: {
    label: 'go',
    line: ['//'], block: ['/*', '*/'], quote: '"\'', template: '`',
    keyword: words('break case chan const continue default defer else fallthrough for func go goto if import '
      + 'interface map package range return select struct switch type var'),
    type: words('bool byte complex64 complex128 error float32 float64 int int8 int16 int32 int64 rune string '
      + 'uint uint8 uint16 uint32 uint64 uintptr any'),
    literal: words('true false nil iota'),
    builtin: words('append cap clear close complex copy delete imag len make max min new panic print println '
      + 'real recover'),
  },

  rust: {
    label: 'rust',
    line: ['//'], block: ['/*', '*/'], quote: '"', triple: false,
    keyword: words('as async await break const continue crate dyn else enum extern fn for if impl in let loop '
      + 'match mod move mut pub ref return self Self static struct super trait type unsafe use where while '
      + 'macro_rules'),
    type: words('bool char f32 f64 i8 i16 i32 i64 i128 isize str String u8 u16 u32 u64 u128 usize Vec Option '
      + 'Result Box Rc Arc HashMap HashSet'),
    literal: words('true false None Some Ok Err'),
    builtin: words('println print format vec panic assert assert_eq todo unimplemented drop clone into iter'),
  },

  java: {
    label: 'java',
    line: ['//'], block: ['/*', '*/'], quote: '"\'', decorator: '@',
    keyword: words('abstract assert break case catch class continue default do else enum extends final '
      + 'finally for goto if implements import instanceof interface native new package private protected '
      + 'public return static strictfp super switch synchronized this throw throws transient try volatile while '
      + 'record sealed var yield'),
    type: words('boolean byte char double float int long short void String Integer Long Double Float Boolean '
      + 'Character Object List Map Set ArrayList HashMap HashSet Optional Stream'),
    literal: words('true false null'),
    builtin: words('System out println printf Math Arrays Collections Objects Streams'),
  },

  kotlin: {
    label: 'kotlin',
    line: ['//'], block: ['/*', '*/'], quote: '"\'', triple: true, decorator: '@',
    keyword: words('abstract actual annotation as break by catch class companion const constructor continue '
      + 'crossinline data delegate do dynamic else enum expect external final finally for fun get if import in '
      + 'infix init inline inner interface internal is lateinit noinline object open operator out override '
      + 'package private protected public reified return sealed set super suspend tailrec this throw try typealias '
      + 'val var vararg when where while'),
    type: words('Any Boolean Byte Char Double Float Int Long Nothing Short String Unit Array List Map Set'),
    literal: words('true false null'),
    builtin: words('println print listOf mapOf setOf mutableListOf require check let also apply run with'),
  },

  csharp: {
    // C# attributes are `[Obsolete]`, not `@Override`, so there is no decorator here.
    label: 'csharp',
    line: ['//'], block: ['/*', '*/'], quote: '"\'', template: '"',
    keyword: words('abstract as async await base break case catch class const continue default delegate do '
      + 'else enum event explicit extern finally fixed for foreach get if implicit in interface internal is '
      + 'lock namespace new operator out override params private protected public readonly record ref return '
      + 'sealed set sizeof stackalloc static struct switch this throw try typeof unchecked unsafe using var '
      + 'virtual void volatile where while yield'),
    type: words('bool byte char decimal double dynamic float int long object sbyte short string uint ulong '
      + 'ushort void Task List Dictionary IEnumerable'),
    literal: words('true false null'),
    builtin: words('Console WriteLine ToString ToArray Add Remove Contains Select Where ToList'),
  },

  c: {
    label: 'c',
    line: ['//'], block: ['/*', '*/'], quote: '"\'', preproc: '#',
    keyword: words(C_KEYWORDS), type: words(C_TYPES),
    literal: words('true false NULL nullptr'),
    builtin: words('printf fprintf sprintf snprintf scanf malloc calloc realloc free memcpy memset strlen strcmp '
      + 'strcpy fopen fclose fread fwrite'),
  },

  ruby: {
    label: 'ruby',
    line: ['#'], quote: '"\'',
    keyword: words('alias and begin break case class def defined? do else elsif end ensure for if in module '
      + 'next nil not or redo rescue retry return self super then undef unless until when while yield require '
      + 'require_relative include extend attr_accessor attr_reader attr_writer lambda proc raise puts'),
    type: words('Array Hash String Symbol Integer Float Struct Module'),
    literal: words('true false nil'),
    builtin: words('puts print each map select reject reduce inject freeze to_s to_i to_a length size join split'),
  },

  php: {
    label: 'php',
    line: ['//', '#'], block: ['/*', '*/'], quote: '"\'', variable: '$',
    keyword: words('abstract and array as break callable case catch class clone const continue declare default '
      + 'do echo else elseif empty endif endwhile extends final finally fn for foreach function global if '
      + 'implements include include_once instanceof interface isset list namespace new or print private '
      + 'protected public readonly require require_once return static switch throw trait try unset use var '
      + 'while xor yield match'),
    type: words('bool boolean int integer float double string array object mixed void iterable callable null'),
    literal: words('true false null TRUE FALSE NULL'),
    builtin: words('count strlen array_map array_filter array_merge implode explode sprintf printf var_dump '
      + 'print_r json_encode json_decode preg_match'),
  },

  perl: {
    label: 'perl',
    line: ['#'], quote: '"\'', variable: '$',
    keyword: words('my our local sub use package require if elsif else unless while until for foreach do return '
      + 'last next redo goto and or not eq ne lt gt le ge cmp defined undef'),
    type: words('int str'),
    literal: words('true false undef'),
    builtin: words('print say printf sprintf chomp chop split join push pop shift unshift keys values each die warn'),
  },

  lua: {
    label: 'lua',
    line: ['--'], quote: '"\'', variable: '',
    keyword: words('and break do else elseif end false for function goto if in local nil not or repeat return '
      + 'then true until while'),
    type: words('string number boolean table function nil'),
    literal: words('true false nil'),
    builtin: words('print pairs ipairs type tostring tonumber require setmetatable rawget rawset pcall error assert'),
  },

  r: {
    label: 'r',
    line: ['#'], quote: '"\'', variable: '',
    keyword: words('if else repeat while function for in next break return library require TRUE FALSE NULL Inf NaN'),
    type: words('numeric character logical integer complex list data.frame matrix vector factor'),
    literal: words('TRUE FALSE NULL NA NA_integer_ Inf NaN'),
    builtin: words('print cat paste paste0 length nrow ncol names subset merge apply lapply sapply vapply mean '
      + 'median sum min max c seq rep'),
  },

  yaml: {
    label: 'yaml',
    line: ['#'], quote: '"\'', props: true,
    literal: words('true false null yes no on off True False Null Yes No On Off ~'),
    keyword: words('--- ...'),
  },

  toml: {
    label: 'toml',
    line: ['#'], quote: '"\'', props: true, triple: true,
    literal: words('true false inf nan'),
    type: words('string integer float boolean datetime array table inline-table local-date local-time '
      + 'offset-date-time'),
  },

  ini: {
    label: 'ini',
    line: ['#', ';'], quote: '"\'', props: true,
    literal: words('true false yes no on off'),
  },

  dockerfile: {
    label: 'dockerfile',
    line: ['#'], quote: '"\'', preproc: '',
    keyword: words('FROM RUN CMD LABEL MAINTAINER EXPOSE ENV ADD COPY ENTRYPOINT VOLUME USER WORKDIR ARG '
      + 'ONBUILD STOPSIGNAL HEALTHCHECK SHELL AS'),
    literal: words('true false'),
    fold: true,
  },

  makefile: {
    label: 'makefile',
    line: ['#'], quote: '"\'', variable: '$',
    keyword: words('ifeq ifneq ifdef ifndef else endif include define endef export unexport override .PHONY '
      + '.SUFFIXES .DEFAULT_GOAL SHELL'),
    builtin: words('echo cd mkdir rm cp mv cat grep sed awk printf touch'),
  },
};

/* Languages whose shape is not "words separated by punctuation" get a scanner.
   Everything else is data in the table above. */
LANGS.html = { label: 'html', scan: scanMarkup };
LANGS.css = { label: 'css', scan: scanCss };
LANGS.markdown = { label: 'markdown', scan: scanMarkdown };
LANGS.diff = { label: 'diff', scan: scanDiff };

/* ── the generic scanner ───────────────────────────────────────────────────── */

const NUMBER = /(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(?:\d[\d_]*)?\.?\d[\d_]*(?:[eE][+-]?\d+)?)[a-zA-Z]*/y;
const PUNCT = '()[]{},;:';

/**
 * Does a string start at `i`? Returns the index of its opening quote, or -1.
 *
 * The prefix branch is what makes `f"..."` one token instead of a call to `f`
 * followed by a string, which is what Python's own highlighter does and the only
 * reason a formatted string reads correctly.
 */
function stringAt(code, i, L) {
  // A template quote is a quote too — it just happens to run across lines.
  const quotes = (L.quote || '') + (L.template || '');
  if (!quotes) return -1;
  if (quotes.includes(code[i])) return i;
  if (L.prefix && L.prefix.includes(code[i])) {
    let j = i + 1;
    while (j < code.length && j - i <= 2 && L.prefix.includes(code[j]) && !quotes.includes(code[j])) j += 1;
    if (j < code.length && quotes.includes(code[j])) return j;
  }
  return -1;
}

/**
 * Consume a string starting at `quoteIndex`. `i` is the token start, so a Python
 * prefix is included in the token.
 *
 * An unterminated string stops at the end of its line rather than swallowing the
 * rest of the file. Highlighters that run to EOF turn one stray quote into a
 * page of string-coloured text, which looks like a bug in the reader rather than
 * in the file.
 */
function readString(code, i, quoteIndex, L) {
  const n = code.length;
  const q = code[quoteIndex];
  const multiline = q === L.template;
  if (L.triple) {
    const triple = q.repeat(3);
    if (code.startsWith(triple, quoteIndex)) {
      const close = code.indexOf(triple, quoteIndex + 3);
      if (close !== -1) return close + 3;
    }
  }
  let j = quoteIndex + 1;
  while (j < n) {
    const ch = code[j];
    if (ch === '\\') { j += 2; continue; }
    if (ch === '\n' && !multiline) return j;
    if (ch === q) return j + 1;
    j += 1;
  }
  return n;
}

function scanCode(code, L) {
  const out = [];
  const n = code.length;
  const kw = L.keyword || EMPTY;
  const ty = L.type || EMPTY;
  const lit = L.literal || EMPTY;
  const bi = L.builtin || EMPTY;
  const quotes = L.quote || '';
  let i = 0;

  while (i < n) {
    const ch = code[i];

    if (ch === ' ' || ch === '\t' || ch === '\r' || ch === '\n') {
      let j = i;
      while (j < n && (code[j] === ' ' || code[j] === '\t' || code[j] === '\r' || code[j] === '\n')) j += 1;
      push(out, code.slice(i, j), 'plain');
      i = j;
      continue;
    }

    // Block comments are tried first: `/*` must not be read as a `/` operator.
    if (L.block && code.startsWith(L.block[0], i)) {
      const close = code.indexOf(L.block[1], i + L.block[0].length);
      if (close !== -1) {
        push(out, code.slice(i, close + L.block[1].length), 'com');
        i = close + L.block[1].length;
        continue;
      }
      // Unclosed: fall through and treat it as an operator, for the same reason
      // an unterminated string stops at the line.
    }

    const lineMark = L.line && L.line.find((mark) => mark && code.startsWith(mark, i));
    if (lineMark) {
      let j = code.indexOf('\n', i);
      if (j === -1) j = n;
      push(out, code.slice(i, j), 'com');
      i = j;
      continue;
    }

    if (L.decorator && ch === L.decorator) {
      let j = i + 1;
      while (j < n && isIdentPart(code[j])) j += 1;
      push(out, code.slice(i, j), 'dec');
      i = j;
      continue;
    }

    // `$HOME`, `$?`, `${x}`, `$(cmd)`, `$1`
    if (L.variable && ch === L.variable) {
      let j = i + 1;
      if (code[j] === '{' || code[j] === '(') {
        const close = code[j] === '{' ? '}' : ')';
        const end = code.indexOf(close, j);
        j = end === -1 ? n : end + 1;
      } else {
        while (j < n && /[\w?*]/.test(code[j])) j += 1;
      }
      push(out, code.slice(i, j), 'var');
      i = j;
      continue;
    }

    // A directive owns its whole line (`#include <stdio.h>`), which is why it is
    // matched here and not as an operator.
    if (L.preproc && ch === L.preproc) {
      let k = i - 1;
      while (k >= 0 && (code[k] === ' ' || code[k] === '\t')) k -= 1;
      if (k < 0 || code[k] === '\n') {
        let j = code.indexOf('\n', i);
        if (j === -1) j = n;
        push(out, code.slice(i, j), 'meta');
        i = j;
        continue;
      }
    }

    const quoteIndex = stringAt(code, i, L);
    if (quoteIndex !== -1) {
      const end = readString(code, i, quoteIndex, L);
      // A key is a string that names something rather than holds something.
      let cls = 'str';
      if (L.props) {
        const k = nextNonSpace(code, end);
        if (code[k] === ':' || code[k] === '=') cls = 'prop';
      }
      push(out, code.slice(i, end), cls);
      i = end;
      continue;
    }

    if (isDigit(ch) || (ch === '.' && isDigit(code[i + 1]))) {
      NUMBER.lastIndex = i;
      const match = NUMBER.exec(code);
      if (match && match[0]) {
        push(out, match[0], 'num');
        i += match[0].length;
        continue;
      }
    }

    if (isIdentStart(ch)) {
      let j = i + 1;
      while (j < n && isIdentPart(code[j])) j += 1;
      const word = code.slice(i, j);
      const key = L.fold ? word.toLowerCase() : word;
      let cls = 'plain';
      if (kw.has(key)) cls = 'kw';
      else if (ty.has(key)) cls = 'typ';
      else if (lit.has(key)) cls = 'lit';
      else if (bi.has(key)) cls = 'fn';
      else if (L.props && [':', '='].includes(code[nextNonSpace(code, j)])) cls = 'prop';
      else if (code[j] === '(') cls = 'fn';           // a call, marked as one
      push(out, word, cls);
      i = j;
      continue;
    }

    if (PUNCT.includes(ch)) {
      push(out, ch, 'pun');
      i += 1;
      continue;
    }

    // Anything else is an operator. Runs are merged by `push`.
    push(out, ch, 'op');
    i += 1;
  }

  return out;
}

/* ── markup ────────────────────────────────────────────────────────────────── */

function scanMarkup(code) {
  const out = [];
  const n = code.length;
  let i = 0;

  while (i < n) {
    if (code.startsWith('<!--', i)) {
      const close = code.indexOf('-->', i + 4);
      const end = close === -1 ? n : close + 3;
      push(out, code.slice(i, end), 'com');
      i = end;
      continue;
    }

    if (code[i] === '<') {
      const next = code[i + 1];
      if (next === '!' || next === '?') {
        const close = code.indexOf('>', i);
        const end = close === -1 ? n : close + 1;
        push(out, code.slice(i, end), 'meta');
        i = end;
        continue;
      }
      push(out, '<', 'pun');
      i += 1;
      if (code[i] === '/') { push(out, '/', 'pun'); i += 1; }

      let j = i;
      while (j < n && /[\w:.-]/.test(code[j])) j += 1;
      push(out, code.slice(i, j), 'tag');
      i = j;

      while (i < n && code[i] !== '>') {
        const ch = code[i];
        if (/\s/.test(ch)) {
          let k = i;
          while (k < n && /\s/.test(code[k])) k += 1;
          push(out, code.slice(i, k), 'plain');
          i = k;
          continue;
        }
        if (ch === '"' || ch === "'") {
          const end = readString(code, i, i, {});
          push(out, code.slice(i, end), 'str');
          i = end;
          continue;
        }
        // {{ interpolation }} in Vue/Angular and ${} in template literals
        if (ch === '{' || ch === '$') {
          const open = code[ch === '{' ? i : i + 1];
          if (open === '{') {
            const close = code.indexOf('}', i);
            const end = close === -1 ? n : close + 1;
            push(out, code.slice(i, end), 'var');
            i = end;
            continue;
          }
        }
        if (ch === '/' || ch === '=') { push(out, ch, 'pun'); i += 1; continue; }
        let k = i;
        while (k < n && /[\w:.-]/.test(code[k])) k += 1;
        if (k === i) { push(out, ch, 'op'); i += 1; continue; }
        push(out, code.slice(i, k), 'attr');
        i = k;
      }
      if (i < n) { push(out, '>', 'pun'); i += 1; }
      continue;
    }

    let j = code.indexOf('<', i);
    if (j === -1) j = n;
    push(out, code.slice(i, j), 'plain');
    i = j;
  }

  return out;
}

/* ── css ───────────────────────────────────────────────────────────────────── */

function scanCss(code) {
  const out = [];
  const n = code.length;
  let depth = 0;
  let i = 0;

  while (i < n) {
    const ch = code[i];

    if (code.startsWith('/*', i)) {
      const close = code.indexOf('*/', i + 2);
      if (close !== -1) { push(out, code.slice(i, close + 2), 'com'); i = close + 2; continue; }
    }

    if (/\s/.test(ch)) {
      let j = i;
      while (j < n && /\s/.test(code[j])) j += 1;
      push(out, code.slice(i, j), 'plain');
      i = j;
      continue;
    }

    if (ch === '"' || ch === "'") {
      const end = readString(code, i, i, {});
      push(out, code.slice(i, end), 'str');
      i = end;
      continue;
    }

    if (ch === '{') { push(out, ch, 'pun'); depth += 1; i += 1; continue; }
    if (ch === '}') { push(out, ch, 'pun'); depth = Math.max(0, depth - 1); i += 1; continue; }

    if (ch === '@') {
      let j = i + 1;
      while (j < n && /[\w-]/.test(code[j])) j += 1;
      push(out, code.slice(i, j), 'meta');
      i = j;
      continue;
    }

    if (ch === '!') {
      let j = i + 1;
      while (j < n && /[\w-]/.test(code[j])) j += 1;
      push(out, code.slice(i, j), 'meta');
      i = j;
      continue;
    }

    // #hex colours and .class / #id selectors share a character; the block depth
    // is what tells them apart.
    if (ch === '#') {
      let j = i + 1;
      while (j < n && /[0-9a-fA-F]/.test(code[j]) && j - i <= 8) j += 1;
      const isColour = j > i + 1 && (j - i === 4 || j - i === 5 || j - i === 7 || j - i === 9);
      push(out, code.slice(i, j), depth > 0 && isColour ? 'num' : 'sel');
      i = j;
      continue;
    }

    if (isDigit(ch) || (ch === '.' && isDigit(code[i + 1]))) {
      NUMBER.lastIndex = i;
      const match = NUMBER.exec(code);
      if (match && match[0]) { push(out, match[0], 'num'); i += match[0].length; continue; }
    }

    if (/[-\w%]/.test(ch)) {
      let j = i;
      while (j < n && /[-\w%]/.test(code[j])) j += 1;
      const word = code.slice(i, j);
      if (depth > 0 && code[nextNonSpace(code, j)] === ':') push(out, word, 'prop');
      else push(out, word, depth > 0 ? 'plain' : 'sel');
      i = j;
      continue;
    }

    if (PUNCT.includes(ch)) { push(out, ch, 'pun'); i += 1; continue; }
    push(out, ch, 'op');
    i += 1;
  }

  return out;
}

/* ── markdown ──────────────────────────────────────────────────────────────── */

const MD_INLINE = /(`+)([^`]*?)\1|(\*\*|__)([\s\S]*?)\3|(\*|_)(?=\S)([^*_]*?[^\s*_])\5|\[([^\]]*)\]\(([^)\s]*)\)/g;

function pushMarkdownInline(out, text) {
  let last = 0;
  MD_INLINE.lastIndex = 0;
  let match = MD_INLINE.exec(text);
  while (match) {
    if (match.index > last) push(out, text.slice(last, match.index), 'plain');
    if (match[1]) {
      push(out, match[1], 'pun');
      push(out, match[2], 'str');
      push(out, match[1], 'pun');
    } else if (match[3]) {
      push(out, match[3], 'pun');
      push(out, match[4], 'typ');
      push(out, match[3], 'pun');
    } else if (match[5]) {
      push(out, match[5], 'pun');
      push(out, match[6], 'typ');
      push(out, match[5], 'pun');
    } else if (match[8] !== undefined) {
      push(out, '[', 'pun');
      push(out, match[7], 'plain');
      push(out, '](', 'pun');
      push(out, match[8], 'str');
      push(out, ')', 'pun');
    }
    last = MD_INLINE.lastIndex;
    match = MD_INLINE.exec(text);
  }
  if (last < text.length) push(out, text.slice(last), 'plain');
}

function scanMarkdown(code) {
  const out = [];
  let fenced = false;

  code.split('\n').forEach((line, index) => {
    if (index) push(out, '\n', 'plain');

    const fence = /^\s*(```|~~~)\s*([^\s`]*)/.exec(line);
    if (fence) {
      push(out, line.slice(0, line.indexOf(fence[1])), 'plain');
      push(out, fence[1], 'meta');
      push(out, line.slice(line.indexOf(fence[1]) + fence[1].length), fenced ? 'plain' : 'lit');
      fenced = !fenced;
      return;
    }
    if (fenced) { push(out, line, 'plain'); return; }

    const heading = /^(\s*)(#{1,6})(\s+.*)?$/.exec(line);
    if (heading) {
      push(out, heading[1], 'plain');
      push(out, heading[2], 'meta');
      push(out, heading[3] || '', 'head');
      return;
    }

    const rule = /^(\s*)([-*_])(\s*(?:\2\s*){2,})$/.exec(line);
    if (rule) { push(out, line, 'meta'); return; }

    const bullet = /^(\s*)([-*+]|\d{1,9}[.)])(\s+)/.exec(line);
    if (bullet) {
      push(out, bullet[1], 'plain');
      push(out, bullet[2], 'lit');
      push(out, bullet[3], 'plain');
      pushMarkdownInline(out, line.slice(bullet[0].length));
      return;
    }

    const quote = /^(\s*>+\s?)/.exec(line);
    if (quote) {
      push(out, quote[1], 'meta');
      pushMarkdownInline(out, line.slice(quote[1].length));
      return;
    }

    const row = /^\s*\|/.exec(line);
    if (row) { push(out, line, line.includes('---') ? 'meta' : 'plain'); return; }

    pushMarkdownInline(out, line);
  });

  return out;
}

/* ── diff ──────────────────────────────────────────────────────────────────── */

function scanDiff(code) {
  const out = [];
  code.split('\n').forEach((line, index) => {
    if (index) push(out, '\n', 'plain');
    if (/^(diff |index |--- |\+\+\+ |@@ )/.test(line)) push(out, line, 'meta');
    else if (line.startsWith('+')) push(out, line, 'add');
    else if (line.startsWith('-')) push(out, line, 'del');
    else push(out, line, 'plain');
  });
  return out;
}

/* ── public API ────────────────────────────────────────────────────────────── */

function tokenize(code, lang) {
  const spec = LANGS[lang];
  if (!spec) return [{ text: code, cls: 'plain' }];
  return spec.scan ? spec.scan(code) : scanCode(code, spec);
}

/**
 * Split `code` into one HTML string per line, with every span closed per line.
 *
 * Per-line output is the whole reason this is not a single HTML blob: the
 * `read_file` tool returns a line-numbered listing, and a caller cannot align a
 * gutter against HTML that a multi-line block comment has left a span open across.
 */
export function highlightLines(code, lang) {
  // Line endings are normalised to \n before anything else looks at the text. `\r` is
  // invisible on screen, but a token that ends in one would sit *after* the `$` of a
  // `/m`-anchored pattern and quietly stop the pattern from matching, so a file written
  // with CRLF would highlight as if it were unindented prose. Normalising once, here,
  // means no scanner has to know that two line endings exist.
  const source = String(code ?? '').replace(/\r\n?/g, '\n');
  const lines = [''];
  for (const token of tokenize(source, lang)) {
    const parts = String(token.text).split('\n');
    for (let i = 0; i < parts.length; i += 1) {
      if (i) lines.push('');
      if (!parts[i]) continue;
      const piece = esc(parts[i]);
      lines[lines.length - 1] += token.cls === 'plain' ? piece : `<span class="t-${token.cls}">${piece}</span>`;
    }
  }
  return lines;
}

export function highlight(code, lang) {
  return highlightLines(code, lang).join('\n');
}

/** The display name for a language id, for a code-block header. */
export function languageLabel(lang) {
  const key = String(lang ?? '');
  // An id we do not know could still be a name the caller read off the fence, and
  // showing `ocaml` is more useful than showing `text` at it.
  return (LANGS[key] && LANGS[key].label) || key || 'text';
}

/* ── naming ────────────────────────────────────────────────────────────────── */

const ALIASES = {
  py: 'python', python: 'python', python3: 'python', pyi: 'python', pyw: 'python',
  js: 'javascript', javascript: 'javascript', node: 'javascript', mjs: 'javascript', cjs: 'javascript',
  jsx: 'javascript', es6: 'javascript',
  ts: 'typescript', typescript: 'typescript', tsx: 'typescript', mts: 'typescript', cts: 'typescript',
  json: 'json', jsonc: 'json', json5: 'json', jsonl: 'json', ndjson: 'json', ipynb: 'json', geojson: 'json',
  sh: 'shell', bash: 'shell', shell: 'shell', zsh: 'shell', ksh: 'shell', dash: 'shell', console: 'shell',
  shellscript: 'shell', envrc: 'shell',
  ps1: 'powershell', powershell: 'powershell', pwsh: 'powershell', psm1: 'powershell',
  sql: 'sql', postgres: 'sql', postgresql: 'sql', mysql: 'sql', sqlite: 'sql', plpgsql: 'sql',
  go: 'go', golang: 'go',
  rs: 'rust', rust: 'rust',
  java: 'java',
  kt: 'kotlin', kotlin: 'kotlin', kts: 'kotlin',
  cs: 'csharp', csharp: 'csharp', 'c#': 'csharp', dotnet: 'csharp',
  c: 'c', h: 'c', cc: 'c', cpp: 'c', 'c++': 'c', cxx: 'c', hpp: 'c', hh: 'c', objc: 'c', m: 'c',
  rb: 'ruby', ruby: 'ruby',
  php: 'php',
  pl: 'perl', perl: 'perl',
  lua: 'lua',
  r: 'r',
  yml: 'yaml', yaml: 'yaml',
  toml: 'toml',
  ini: 'ini', cfg: 'ini', conf: 'ini', editorconfig: 'ini', properties: 'ini', env: 'ini',
  dockerfile: 'dockerfile', docker: 'dockerfile', containerfile: 'dockerfile',
  makefile: 'makefile', make: 'makefile', mk: 'makefile', cmake: 'makefile',
  html: 'html', htm: 'html', xml: 'html', svg: 'html', vue: 'html', xhtml: 'html', svelte: 'html',
  css: 'css', scss: 'css', sass: 'css', less: 'css',
  md: 'markdown', markdown: 'markdown', mdx: 'markdown',
  diff: 'diff', patch: 'diff', udiff: 'diff',
  text: '', txt: '', plain: '', plaintext: '', log: '', none: '',
};

/** Files with no extension that are still worth naming, mirroring the backend's
 *  `TEXT_NAMES` so a chip and the model agree about what a file is. */
const BY_NAME = {
  dockerfile: 'dockerfile', containerfile: 'dockerfile',
  makefile: 'makefile', gnumakefile: 'makefile',
  gemfile: 'ruby', rakefile: 'ruby', vagrantfile: 'ruby', brewfile: 'ruby',
  procfile: 'shell', bashrc: 'shell', zshrc: 'shell', bash_profile: 'shell', profile: 'shell',
  '.gitignore': 'ini', '.dockerignore': 'ini', '.npmignore': 'ini', '.gitattributes': 'ini',
  '.editorconfig': 'ini', '.npmrc': 'ini', '.env': 'ini', '.flake8': 'ini', '.babelrc': 'json',
  'package.json': 'json', 'tsconfig.json': 'json', 'requirements.txt': '', '.prettierrc': 'json',
  'go.mod': 'go',
};

/** Suffixes that are text but not code — a chip, but no highlighting to fake. */
const PLAIN_SUFFIXES = new Set(['txt', 'text', 'log', 'csv', 'tsv', 'rst', 'org', 'adoc',
  'readme', 'license', 'changelog', 'authors', 'notice']);

function suffixOf(name) {
  const base = String(name || '').split(/[\\/]/).pop().toLowerCase();
  if (!base) return '';
  const dot = base.lastIndexOf('.');
  if (dot <= 0) return '';
  return base.slice(dot + 1);
}

/** The language for a filename, or '' when it is not one we highlight. */
export function languageForName(name) {
  const base = String(name || '').split(/[\\/]/).pop().toLowerCase();
  if (!base) return '';
  if (Object.prototype.hasOwnProperty.call(BY_NAME, base)) return BY_NAME[base];
  const suffix = suffixOf(base);
  if (!suffix) return '';
  return Object.prototype.hasOwnProperty.call(ALIASES, suffix) ? ALIASES[suffix] : '';
}

/** Is this a file whose contents are text, whatever the language? Drives the chip. */
export function isTextName(name) {
  const base = String(name || '').split(/[\\/]/).pop().toLowerCase();
  if (!base) return false;
  if (Object.prototype.hasOwnProperty.call(BY_NAME, base)) return true;
  if (PLAIN_SUFFIXES.has(base)) return true;
  const suffix = suffixOf(base);
  if (!suffix) return false;
  return PLAIN_SUFFIXES.has(suffix) || Object.prototype.hasOwnProperty.call(ALIASES, suffix);
}

/** Resolve a fence's info string — `python`, `py`, `js`, or `js title="x"`. */
export function languageForHint(hint) {
  // Fence markers are stripped rather than rejected: an info string can reach us with
  // its own backticks still attached when a block was quoted inside another block.
  const raw = String(hint || '').trim().toLowerCase().replace(/^[`~]+/, '').replace(/[`~]+$/, '');
  if (!raw) return '';
  const first = raw.split(/[\s,{;]/)[0];
  for (const key of [raw, first]) {
    if (Object.prototype.hasOwnProperty.call(ALIASES, key)) return ALIASES[key];
  }
  // A fence may name the file instead of the language: ```app.py
  return languageForName(first);
}

/* ── auto-detection ────────────────────────────────────────────────────────── */

/**
 * Guess the language of a block whose fence said nothing useful.
 *
 * A scorer rather than a chain of early returns, so that a weak signal in one
 * language cannot mask a strong one in another — and so that the answer is
 * order-independent and reproducible. Ties go to whichever candidate was scored
 * first, which is why the list below is ordered by specificity.
 *
 * Below the threshold it returns '', and the block renders as plain text. A wrong
 * guess is worse than no guess: mis-coloured code reads as corruption.
 */
export function detectLanguage(code) {
  const text = String(code || '');
  if (!text.trim()) return '';

  const lines = text.split('\n').slice(0, 40);
  const head = lines.join('\n');
  const found = [];
  const add = (lang, score) => { if (score > 0) found.push({ lang, score }); };

  const trimmed = text.trim();
  if (trimmed[0] === '{' || trimmed[0] === '[') {
    try { JSON.parse(trimmed); add('json', 10); } catch { /* not JSON after all */ }
  }

  if (/^@@ .* @@/m.test(head)) add('diff', 9);
  if (/^\+\+\+ /m.test(head) && /^--- /m.test(head)) add('diff', 8);

  if (/^\s*<!doctype\s+html/i.test(text)) add('html', 10);
  if (/^\s*<\?xml/i.test(text)) add('html', 10);
  if (/<(html|head|body|div|span|p|a|script|style|table|ul|ol|li|nav|section|button|input)\b[^>]*>/i.test(text)) {
    add('html', 6);
  }
  if ((text.match(/<\/[a-zA-Z][\w:-]*>/g) || []).length >= 3) add('html', 5);

  let md = 0;
  if (/^\s{0,3}#{1,6}\s+\S/m.test(text)) md += 3;
  if (/^\s*```/m.test(text)) md += 3;
  if (/\[[^\]]+\]\([^)\s]+\)/.test(text)) md += 2;
  if (/^\s*[-*+]\s+\S/m.test(text)) md += 1;
  if (/^\s*>\s+\S/m.test(text)) md += 1;
  if (md) add('markdown', Math.min(md, 9));

  if (/\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b/i.test(text)
    && /\b(FROM|INTO|TABLE|SET|WHERE|VALUES|JOIN)\b/i.test(text)) add('sql', 8);

  if (/^#!.*\bpython[0-9.]*\b/.test(text)) add('python', 10);
  if (/^#!.*\bnode\b/.test(text)) add('javascript', 10);
  if (/^#!.*\b(bash|sh|zsh|ksh|dash)\b/.test(text)) add('shell', 10);

  let py = 0;
  if (/^\s*def\s+\w+\s*\([^)]*\)\s*(->[^:]+)?:/m.test(text)) py += 4;
  if (/^\s*class\s+\w+(\([^)]*\))?\s*:/m.test(text)) py += 3;
  if (/^\s*(from\s+[\w.]+\s+import|import\s+[\w.]+)/m.test(text)) py += 3;
  if (/^\s*elif\b/m.test(text)) py += 2;
  if (/\bself\b/.test(text)) py += 2;
  if (/\b(None|True|False)\b/.test(text)) py += 1;
  if (/^\s*@\w+/m.test(text)) py += 1;
  if (/"""|'''/.test(text)) py += 1;
  if (/\bprint\s*\(/.test(text)) py += 1;
  if (py) add('python', Math.min(py, 10));

  let js = 0;
  if (/\bconst\s+\w+\s*=/.test(text) || /\blet\s+\w+\s*=/.test(text) || /\bvar\s+\w+\s*=/.test(text)) js += 3;
  if (/=>/.test(text)) js += 2;
  if (/\bfunction\s*[\w(]/.test(text)) js += 2;
  if (/\bconsole\.(log|error|warn)\(/.test(text)) js += 4;
  if (/\b(require\(|module\.exports|export\s+(default|const|function))/.test(text)) js += 3;
  if (/\b(document|window|localStorage)\.\w+/.test(text)) js += 2;
  if (/\bnew\s+[A-Z]\w*\s*\(/.test(text)) js += 1;
  if (js) add('javascript', Math.min(js, 10));

  let ts = 0;
  if (/\binterface\s+\w+\s*[{<]/.test(text)) ts += 4;
  if (/:\s*(string|number|boolean|void|unknown)\b/.test(text)) ts += 3;
  if (/\btype\s+\w+\s*=/.test(text)) ts += 3;
  if (/\b(enum|namespace|readonly|implements)\b/.test(text)) ts += 2;
  if (/\bas\s+(const|string|number|unknown)\b/.test(text)) ts += 1;
  // A lone `readonly` or `implements` is not enough: both turn up as ordinary words
  // in JavaScript, and the +2 bonus would carry a 2 to the threshold on its own.
  if (ts >= 3) add('typescript', Math.min(ts + 2, 10));

  let sh = 0;
  if (/^\s*\$\s+\w/m.test(text)) sh += 5;
  if (/^\s*(fi|done|esac)\s*$/m.test(text)) sh += 4;
  if (/\$\{?\w+\}?/.test(text) && /\b(echo|export|cd|mkdir|rm|grep|sed|awk|cat)\b/.test(text)) sh += 4;
  if (/^\s*if\s+\[\[?/m.test(text)) sh += 3;
  if (sh) add('shell', Math.min(sh, 9));

  let yaml = 0;
  if (/^---\s*$/m.test(text)) yaml += 4;
  if (/^\s*[\w."'-]+:\s*(\S.*)?$/m.test(text)) yaml += 3;
  if (/^\s*-\s+\S/m.test(text)) yaml += 1;
  if (/^\s*#/m.test(text)) yaml += 1;
  if (yaml && !/[;{}]|^\s*(def|function|const|let|import)\b/m.test(text)) add('yaml', Math.min(yaml, 6));

  if (/^\s*\[[A-Za-z_][\w.-]*\]\s*$/m.test(text) && /^\s*[\w.-]+\s*=/.test(text)) add('toml', 7);

  if (/^\s*FROM\s+\S+/im.test(text) && /^\s*(RUN|CMD|COPY|ADD|ENV|WORKDIR|ENTRYPOINT|EXPOSE)\b/im.test(text)) {
    add('dockerfile', 9);
  }
  if (/^[A-Za-z_][\w.-]*\s*:(?!=)/m.test(text) && /^\t\S/m.test(text)) add('makefile', 7);

  if (/^\s*func\s+(\(\w+\s+\*?\w+\)\s*)?\w+\s*\(/m.test(text) || /^\s*package\s+\w+\s*$/m.test(text)) add('go', 9);
  if (/^\s*(pub\s+)?fn\s+\w+/m.test(text) || /\blet\s+mut\b/.test(text) || /\buse\s+std::/.test(text)) {
    add('rust', 9);
  }
  if (/^\s*#include\s*[<"]/m.test(text)) add('c', 9);
  if (/\bpublic\s+(static\s+)?void\s+main\s*\(/.test(text)) add('java', 8);
  if (/\b(int|void|char|double|float|struct|unsigned)\s+\w+\s*[;=(]/m.test(text)) add('c', 5);

  if (/^\s*def\s+\w+[\s(]/m.test(text) && /^\s*end\s*$/m.test(text)) add('ruby', 8);
  if (/<\?php/.test(text)) add('php', 10);
  if (/\busing\s+System(\.\w+)*\s*;/i.test(text) && /\b(namespace|class|public|private|static)\b/.test(text)) {
    add('csharp', 8);
  }
  if (/\bfun\s+\w+\s*\(/.test(text) || (/\bval\s+\w+\s*=/.test(text) && /\bprintln\s*\(/.test(text))) {
    add('kotlin', 8);
  }

  let css = 0;
  if (/\.[\w-]+\s*\{[^}]*[\w-]+\s*:[^;]+;/m.test(text)) css += 5;
  if (/^\s*[.#][\w-]+(\s*[,>+~]\s*[.#]?[\w-]+)*\s*\{/m.test(text)) css += 4;
  if (/@(media|import|supports)\b/.test(text)) css += 3;
  if (css) add('css', Math.min(css, 8));

  let best = null;
  for (const entry of found) if (!best || entry.score > best.score) best = entry;
  return best && best.score >= 4 ? best.lang : '';
}

/** The language to use for a block: what the fence said, else what it looks like. */
export function resolveLanguage(code, hint) {
  const fromHint = languageForHint(hint);
  if (fromHint) return fromHint;
  if (String(hint || '').trim()) return '';        // the fence named a plain language
  return detectLanguage(code);
}
