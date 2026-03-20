# Frontend Changes: Light/Dark Mode Toggle Button

## Feature
Added a light/dark mode toggle button positioned fixed in the top-right corner of the UI.

## Files Modified

### `frontend/index.html`
- Added a `<button class="theme-toggle" id="themeToggle">` element just inside `<body>`, before `.container`
- Button contains two SVG icons: a sun (shown in dark mode) and a moon (shown in light mode)
- Includes `aria-label` and `title` attributes for accessibility

### `frontend/style.css`
- Added `body.light-theme` CSS variable overrides for light mode colors (background, surface, text, borders)
- Added `body { transition: background-color 0.3s ease, color 0.3s ease; }` for smooth theme switching
- Added `.theme-toggle` button styles: fixed position top-right, circular 40×40px, matches existing design language
- Added hover/focus/active states consistent with existing interactive elements (uses `--focus-ring`, `--primary-color`, etc.)
- Added `.icon-sun` / `.icon-moon` icon rules with smooth opacity + rotation transitions when toggling

### `frontend/script.js`
- Added `initThemeToggle()` function called on `DOMContentLoaded`
- Reads saved preference from `localStorage` on page load and applies it
- Toggles `light-theme` class on `<body>` and persists choice to `localStorage` key `"theme"`

## Design Decisions
- Button is `position: fixed` so it stays visible regardless of scroll or layout
- Uses existing CSS custom properties (`--surface`, `--border-color`, `--primary-color`, `--focus-ring`) to stay consistent with the design system
- Icon swap uses CSS opacity + rotate transforms for a smooth animated transition (no JS DOM manipulation needed)
- `localStorage` persists the user's preference across page reloads; defaults to dark theme

## Light Theme Color Palette
| Variable | Light value |
|---|---|
| `--background` | `#f8fafc` |
| `--surface` | `#ffffff` |
| `--surface-hover` | `#e2e8f0` |
| `--text-primary` | `#0f172a` |
| `--text-secondary` | `#64748b` |
| `--border-color` | `#e2e8f0` |
| `--assistant-message` | `#f1f5f9` |