# cadette.org — landing page

A self-contained static landing page for [Cadette](https://github.com/chasegan/cadette).
No build step, no dependencies. Just `index.html`, `styles.css`, and `favicon.svg`.

## Files

| File          | Purpose                                                        |
|---------------|---------------------------------------------------------------|
| `index.html`  | The page.                                                     |
| `styles.css`  | All styling. Palette + type from the Cadette Brand Guide.     |
| `favicon.svg` | C-monogram favicon.                                           |
| `CNAME`       | Custom domain for GitHub Pages (`cadette.org`). Remove if hosting elsewhere. |

## Preview locally

```sh
cd docs
python3 -m http.server 8000
# open http://localhost:8000
```

## Download buttons

The macOS button points at a **stable** GitHub "latest release" URL:

```
https://github.com/chasegan/cadette/releases/latest/download/Cadette-macOS.dmg
```

This always redirects to the newest release **as long as the release asset is
named exactly `Cadette-macOS.dmg`** (no version number in the filename). Make sure
your release workflow publishes the asset under that stable name. To change the
platform filenames, edit the `href`s in `index.html`.

Windows and Linux are shown as disabled "soon" stubs. When builds exist, replace
each stub `<span>` with an `<a class="btn …">` pointing at, e.g.:

```
https://github.com/chasegan/cadette/releases/latest/download/Cadette-Windows.exe
https://github.com/chasegan/cadette/releases/latest/download/Cadette-Linux.AppImage
```

## Deploy on GitHub Pages

1. These files live in `/docs` of the `chasegan/cadette` repo.
2. Repo **Settings → Pages** → Source = "Deploy from a branch", branch `main`, folder `/docs`.
3. Set the custom domain to `cadette.org` (the `CNAME` file already does this).
4. In **GoDaddy DNS** add:
   - Four `A` records for the apex `@` → `185.199.108.153`, `185.199.109.153`,
     `185.199.110.153`, `185.199.111.153`
   - One `CNAME` for `www` → `<your-github-username>.github.io`
5. Wait for DNS + the free TLS cert to provision (minutes to a couple of hours),
   then enable **Enforce HTTPS** in the Pages settings.

## Deploy elsewhere

It's plain static files — drop the folder into Cloudflare Pages, Netlify, or any
static host. Delete `CNAME` (that's GitHub-Pages-specific) and configure the
domain in that host instead.
