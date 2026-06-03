# GitHub Pages Deployment

The Systematic Equity Research Platform showcase site is a static, no-build GitHub Pages site in `docs/`.

## Recommended Configuration

1. Push the repository to GitHub.
2. Open the repository on GitHub.
3. Go to `Settings` -> `Pages`.
4. Under `Build and deployment`, choose:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/docs`
5. Save the configuration.

GitHub Pages will serve `docs/index.html` automatically.

## Expected URL

For the current remote:

```text
https://johnA750.github.io/Trading-Backtester/
```

## Local Preview

From the repository root:

```powershell
python -m http.server 8000 -d docs
```

Then open:

```text
http://localhost:8000
```

## Generated Site Files

- `docs/index.html` - project showcase page
- `docs/styles.css` - responsive visual styling
- `docs/site.js` - lightweight scroll enhancement
- `docs/.nojekyll` - disables Jekyll processing
- `docs/assets/*.png` - curated dashboard screenshots

## Notes

- No backend or build step is required.
- The site uses relative asset paths, so it works from the `/docs` GitHub Pages source.
- If the repository is renamed, update the expected URL and any repository links in `docs/index.html`.
