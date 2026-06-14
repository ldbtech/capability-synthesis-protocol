import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Frontend on 5174 (CSV-RAG uses 5173). Proxies /api to the algoviz backend :8001.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/api": "http://localhost:8001",
    },
  },
});
