---
hide:
  - toc
---

# HTTP API reference

Generated from the service's own routes at build time. For a tour with
curl, see the [HTTP guide](../getting-started/http.md). A running server
serves the same spec at `/openapi.json` and an interactive version at
`/docs`.

<div id="redoc-container" class="tx-redoc"></div>
<script src="https://cdn.jsdelivr.net/npm/redoc@2.1.5/bundles/redoc.standalone.js"></script>
<script>
  Redoc.init("../openapi.json", {
    hideDownloadButton: false,
    expandResponses: "200",
    theme: { colors: { primary: { main: "#6366F1" } },
             typography: { fontFamily: "Inter, sans-serif", code: { fontFamily: "JetBrains Mono, monospace" } } }
  }, document.getElementById("redoc-container"));
</script>
