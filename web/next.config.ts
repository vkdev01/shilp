import type { NextConfig } from "next";

// `next dev` proxies the API to a running `shilp serve`; `next build` exports
// plain files that the Python server hands out itself.
const api = process.env.SHILP_API ?? "http://127.0.0.1:8000";
const proxied = ["/api/:path*", "/v1/:path*", "/health", "/metrics", "/docs", "/openapi.json"];

const config: NextConfig =
  process.env.NODE_ENV === "development"
    ? {
        trailingSlash: true,
        agentRules: false,
        // FastAPI routes have no trailing slash; do not let Next add one before proxying.
        skipTrailingSlashRedirect: true,
        async rewrites() {
          return proxied.map((source) => ({ source, destination: `${api}${source}` }));
        },
      }
    : {
        output: "export",
        trailingSlash: true,
        images: { unoptimized: true },
      };

export default config;
