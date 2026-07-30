# ERA V5 — Web App Templates

Two ready-to-copy templates for new session web apps.

---

## 📁 Structure

```
templates/
├── dark/           # Session 5 dark theme (base: dark, toggle to light)
│   ├── index.html  # HTML scaffold with navigation + hero + sections
│   └── style.css   # CSS with dark :root tokens + [data-theme="light"] override
│
└── light/          # Session 4 light theme (base: light, toggle to dark)
    ├── index.html  # HTML scaffold with panel-switching nav + mobile topbar
    └── style.css   # CSS with light :root tokens + [data-theme="dark"] override
```

---

## 🎨 When to use which template?

| Template | Default look | Best for |
|----------|-------------|----------|
| **`dark/`** | Deep space dark (S5 style) | Dense data visualisations, donut charts, heavy tables |
| **`light/`** | Clean slate white (S4 style) | Multi-panel dashboards, chart.js charts, pipeline flows |

Both templates include a **🌙 / ☀️ toggle button** (top-right corner) that switches between dark and light and **saves the preference** in `localStorage`.

---

## 🚀 How to start a new session webapp

1. **Copy** the appropriate template folder into your session directory:
   ```
   cp -r templates/dark/ v5/session6/webapp/
   ```

2. **Edit `index.html`**: Replace all `[PLACEHOLDER]` tokens with real session content.

3. **Create `data.js`**: Export your data as `window.SESSION_DATA = { ... }`.

4. **Create `app.js`**: Import `window.SESSION_DATA` and build your widgets.

5. **Extend `style.css`**: Append section-specific layout classes below the base tokens — do not edit the token block.

---

## 🔄 Theme Toggle — How it works

Both templates use a single self-contained IIFE in the HTML. No separate JS file needed.

```html
<button class="theme-toggle" id="themeToggle">
  <span class="theme-toggle-icon" id="themeIcon">☀️</span>
  <span class="theme-toggle-label" id="themeLabel">Light</span>
</button>
```

```js
(function () {
  // reads localStorage, sets data-theme attr on <html>,
  // flips icon + label on click, writes back to localStorage
})();
```

The CSS uses **`[data-theme="..."] { --var: value; }`** overrides — so every component that uses CSS vars automatically adapts.

---

## ✅ Live session webapps using this system

| Session | Template base | Toggle default |
|---------|--------------|----------------|
| Session 4 (`v5/session4/webapp/`) | light | Light → Dark |
| Session 5 (`v5/session5/webapp/`) | dark  | Dark → Light  |
