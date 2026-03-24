import { useState, useMemo } from 'react'
import { useTheme } from '../context/ThemeContext'

interface CodeBlockProps {
  code: string
  language?: string
  title?: string
}

/* ---------- Token types for syntax highlighting ---------- */
type TokenType = 'comment' | 'keyword' | 'string' | 'number' | 'function' | 'operator' | 'punctuation' | 'key' | 'boolean' | 'decorator' | 'text'

interface Token {
  type: TokenType
  value: string
}

/* ---------- Colors (dark mode) ---------- */
const darkColors: Record<TokenType, string> = {
  comment: '#6b7280',
  keyword: '#c084fc',
  string: '#34d399',
  number: '#f59e0b',
  function: '#60a5fa',
  operator: '#9ca3af',
  punctuation: '#9ca3af',
  key: '#60a5fa',
  boolean: '#f59e0b',
  decorator: '#c084fc',
  text: '#d1d5db',
}

const lightColors: Record<TokenType, string> = {
  comment: '#9ca3af',
  keyword: '#7c3aed',
  string: '#059669',
  number: '#d97706',
  function: '#2563eb',
  operator: '#64748b',
  punctuation: '#64748b',
  key: '#2563eb',
  boolean: '#d97706',
  decorator: '#7c3aed',
  text: '#374151',
}

/* ---------- Tokenizers ---------- */
function tokenizePython(code: string): Token[] {
  const tokens: Token[] = []
  const keywords = new Set([
    'import', 'from', 'as', 'def', 'return', 'class', 'if', 'else', 'elif',
    'for', 'while', 'with', 'try', 'except', 'raise', 'True', 'False', 'None',
    'print', 'self', 'in', 'not', 'and', 'or', 'is', 'lambda', 'yield',
    'async', 'await', 'pass', 'break', 'continue', 'del', 'global', 'nonlocal',
    'assert', 'finally',
  ])
  let i = 0
  while (i < code.length) {
    // Comments
    if (code[i] === '#') {
      let end = code.indexOf('\n', i)
      if (end === -1) end = code.length
      tokens.push({ type: 'comment', value: code.slice(i, end) })
      i = end
      continue
    }
    // Triple-quoted strings
    if (code.slice(i, i + 3) === '"""' || code.slice(i, i + 3) === "'''") {
      const quote = code.slice(i, i + 3)
      let end = code.indexOf(quote, i + 3)
      if (end === -1) end = code.length - 3
      tokens.push({ type: 'string', value: code.slice(i, end + 3) })
      i = end + 3
      continue
    }
    // Strings
    if (code[i] === '"' || code[i] === "'") {
      const quote = code[i]
      let j = i + 1
      while (j < code.length && code[j] !== quote) {
        if (code[j] === '\\') j++
        j++
      }
      tokens.push({ type: 'string', value: code.slice(i, j + 1) })
      i = j + 1
      continue
    }
    // Decorators
    if (code[i] === '@' && (i === 0 || code[i - 1] === '\n' || /\s/.test(code[i - 1]))) {
      let j = i + 1
      while (j < code.length && /[\w.]/.test(code[j])) j++
      tokens.push({ type: 'decorator', value: code.slice(i, j) })
      i = j
      continue
    }
    // Numbers
    if (/\d/.test(code[i]) && (i === 0 || !/[\w.]/.test(code[i - 1]))) {
      let j = i
      while (j < code.length && /[\d.eE_xXa-fA-F]/.test(code[j])) j++
      tokens.push({ type: 'number', value: code.slice(i, j) })
      i = j
      continue
    }
    // Words (keywords, functions, identifiers)
    if (/[a-zA-Z_]/.test(code[i])) {
      let j = i
      while (j < code.length && /[\w]/.test(code[j])) j++
      const word = code.slice(i, j)
      if (keywords.has(word)) {
        tokens.push({ type: 'keyword', value: word })
      } else if (j < code.length && code[j] === '(') {
        tokens.push({ type: 'function', value: word })
      } else {
        tokens.push({ type: 'text', value: word })
      }
      i = j
      continue
    }
    // Operators & punctuation
    if (/[=+\-*/<>!&|^~%]/.test(code[i])) {
      tokens.push({ type: 'operator', value: code[i] })
      i++
      continue
    }
    if (/[()[\]{},;:.]/.test(code[i])) {
      tokens.push({ type: 'punctuation', value: code[i] })
      i++
      continue
    }
    // Whitespace and other
    tokens.push({ type: 'text', value: code[i] })
    i++
  }
  return tokens
}

function tokenizeBash(code: string): Token[] {
  const tokens: Token[] = []
  const keywords = new Set([
    'python', 'pip', 'git', 'cd', 'source', 'export', 'npm', 'pytest',
    'ruff', 'mypy', 'CUDA_VISIBLE_DEVICES', 'accelerate', 'huggingface-cli',
    'rm', 'tail', 'mkdir', 'echo', 'cat', 'chmod', 'sudo', 'apt', 'brew',
  ])
  let i = 0
  while (i < code.length) {
    // Comments
    if (code[i] === '#') {
      let end = code.indexOf('\n', i)
      if (end === -1) end = code.length
      tokens.push({ type: 'comment', value: code.slice(i, end) })
      i = end
      continue
    }
    // Strings
    if (code[i] === '"' || code[i] === "'") {
      const quote = code[i]
      let j = i + 1
      while (j < code.length && code[j] !== quote) {
        if (code[j] === '\\') j++
        j++
      }
      tokens.push({ type: 'string', value: code.slice(i, j + 1) })
      i = j + 1
      continue
    }
    // Flags (--flag or -f)
    if (code[i] === '-' && i + 1 < code.length && /[a-zA-Z-]/.test(code[i + 1])) {
      let j = i
      while (j < code.length && /[\w-]/.test(code[j])) j++
      tokens.push({ type: 'function', value: code.slice(i, j) })
      i = j
      continue
    }
    // Words
    if (/[a-zA-Z_]/.test(code[i])) {
      let j = i
      while (j < code.length && /[\w.-]/.test(code[j])) j++
      const word = code.slice(i, j)
      if (keywords.has(word)) {
        tokens.push({ type: 'keyword', value: word })
      } else {
        tokens.push({ type: 'text', value: word })
      }
      i = j
      continue
    }
    // Numbers
    if (/\d/.test(code[i])) {
      let j = i
      while (j < code.length && /[\d.]/.test(code[j])) j++
      tokens.push({ type: 'number', value: code.slice(i, j) })
      i = j
      continue
    }
    tokens.push({ type: 'text', value: code[i] })
    i++
  }
  return tokens
}

function tokenizeYaml(code: string): Token[] {
  const tokens: Token[] = []
  const lines = code.split('\n')
  lines.forEach((line, lineIdx) => {
    if (lineIdx > 0) tokens.push({ type: 'text', value: '\n' })
    let i = 0
    // Leading whitespace
    while (i < line.length && /\s/.test(line[i])) {
      tokens.push({ type: 'text', value: line[i] })
      i++
    }
    // Comment line
    if (line[i] === '#') {
      tokens.push({ type: 'comment', value: line.slice(i) })
      return
    }
    // Key: value line
    const colonIdx = line.indexOf(':', i)
    if (colonIdx > i && /^[\w.-]+$/.test(line.slice(i, colonIdx).trim())) {
      tokens.push({ type: 'key', value: line.slice(i, colonIdx) })
      tokens.push({ type: 'punctuation', value: ':' })
      const rest = line.slice(colonIdx + 1)
      if (rest.trim()) {
        // Check for inline comment
        const commentIdx = rest.indexOf('#')
        let valuePart = rest
        let commentPart = ''
        if (commentIdx > 0 && rest[commentIdx - 1] === ' ') {
          valuePart = rest.slice(0, commentIdx)
          commentPart = rest.slice(commentIdx)
        }
        const trimmed = valuePart.trim()
        if (/^".*"$/.test(trimmed) || /^'.*'$/.test(trimmed)) {
          tokens.push({ type: 'text', value: valuePart.slice(0, valuePart.indexOf(trimmed)) })
          tokens.push({ type: 'string', value: trimmed })
        } else if (/^(true|false|null|none|yes|no)$/i.test(trimmed)) {
          tokens.push({ type: 'text', value: valuePart.slice(0, valuePart.indexOf(trimmed)) })
          tokens.push({ type: 'boolean', value: trimmed })
        } else if (/^-?\d+\.?\d*(e[+-]?\d+)?$/.test(trimmed)) {
          tokens.push({ type: 'text', value: valuePart.slice(0, valuePart.indexOf(trimmed)) })
          tokens.push({ type: 'number', value: trimmed })
        } else {
          tokens.push({ type: 'text', value: valuePart })
        }
        if (commentPart) {
          tokens.push({ type: 'comment', value: commentPart })
        }
      }
      return
    }
    // List item
    if (line[i] === '-') {
      tokens.push({ type: 'punctuation', value: '-' })
      tokens.push({ type: 'text', value: line.slice(i + 1) })
      return
    }
    tokens.push({ type: 'text', value: line.slice(i) })
  })
  return tokens
}

function tokenizeBibtex(code: string): Token[] {
  const tokens: Token[] = []
  let i = 0
  while (i < code.length) {
    // @ type
    if (code[i] === '@') {
      let j = i + 1
      while (j < code.length && /\w/.test(code[j])) j++
      tokens.push({ type: 'decorator', value: code.slice(i, j) })
      i = j
      continue
    }
    // Braced content
    if (code[i] === '{' || code[i] === '}') {
      tokens.push({ type: 'punctuation', value: code[i] })
      i++
      continue
    }
    // Key = value
    if (/[a-zA-Z]/.test(code[i])) {
      let j = i
      while (j < code.length && /[\w]/.test(code[j])) j++
      const word = code.slice(i, j)
      // Check if followed by =
      let k = j
      while (k < code.length && code[k] === ' ') k++
      if (code[k] === '=') {
        tokens.push({ type: 'key', value: word })
      } else {
        tokens.push({ type: 'string', value: word })
      }
      i = j
      continue
    }
    tokens.push({ type: 'text', value: code[i] })
    i++
  }
  return tokens
}

function tokenizeGeneric(code: string): Token[] {
  return [{ type: 'text', value: code }]
}

function tokenize(code: string, language: string): Token[] {
  switch (language) {
    case 'python':
    case 'py':
      return tokenizePython(code)
    case 'bash':
    case 'shell':
    case 'sh':
      return tokenizeBash(code)
    case 'yaml':
    case 'yml':
      return tokenizeYaml(code)
    case 'bibtex':
      return tokenizeBibtex(code)
    default:
      return tokenizeGeneric(code)
  }
}

export default function CodeBlock({ code, language = 'bash', title }: CodeBlockProps) {
  const { isDark } = useTheme()
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const colors = isDark ? darkColors : lightColors

  const highlighted = useMemo(() => {
    const tokens = tokenize(code, language)
    return tokens.map((token, i) => {
      const color = colors[token.type]
      if (token.type === 'text') {
        return <span key={i}>{token.value}</span>
      }
      return <span key={i} style={{ color }}>{token.value}</span>
    })
  }, [code, language, colors])

  return (
    <div className={`code-block group rounded-xl overflow-hidden transition-all duration-300 ${
      isDark
        ? 'bg-[#0a0f1e] border border-white/[0.06] hover:border-electric-500/20 shadow-lg shadow-black/20'
        : 'bg-[#f7f8fc] border border-navy-200/30 hover:border-navy-300/50 shadow-sm'
    }`}>
      {title && (
        <div className={`flex items-center justify-between px-4 py-2.5 border-b ${
          isDark ? 'border-white/[0.06] bg-white/[0.02]' : 'border-navy-200/20 bg-navy-50/30'
        }`}>
          <div className="flex items-center gap-2.5">
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-[#ff5f57]/80" />
              <div className="w-2.5 h-2.5 rounded-full bg-[#febc2e]/80" />
              <div className="w-2.5 h-2.5 rounded-full bg-[#28c840]/80" />
            </div>
            <span className={`text-xs font-mono ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
              {title}
            </span>
          </div>
          <button
            onClick={handleCopy}
            className={`copy-btn text-xs font-mono px-2.5 py-1 rounded-md transition-all duration-200 ${
              copied
                ? isDark ? 'text-green-400 bg-green-500/10' : 'text-green-600 bg-green-500/10'
                : isDark ? 'text-gray-600 hover:text-gray-300 hover:bg-white/5' : 'text-gray-400 hover:text-gray-600 hover:bg-navy-100'
            }`}
          >
            {copied ? 'Copied!' : 'Copy'}
          </button>
        </div>
      )}
      {!title && (
        <button
          onClick={handleCopy}
          className={`copy-btn absolute top-2.5 right-2.5 text-xs font-mono px-2.5 py-1 rounded-md z-10 transition-all duration-200 opacity-0 group-hover:opacity-100 ${
            copied
              ? isDark ? 'text-green-400 bg-green-500/10' : 'text-green-600 bg-green-500/10'
              : isDark ? 'text-gray-600 hover:text-gray-300 hover:bg-white/10' : 'text-gray-400 hover:text-gray-600 hover:bg-navy-100'
          }`}
        >
          {copied ? 'Copied!' : 'Copy'}
        </button>
      )}
      <pre className={`p-4 overflow-x-auto text-[13px] leading-relaxed ${
        isDark ? 'text-gray-300' : 'text-gray-700'
      }`}>
        <code>{highlighted}</code>
      </pre>
    </div>
  )
}
