import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 后端地址。默认值和 `scripts/api_server.py` 的默认一致。
// 换端口:改这里,或者设环境变量 VITE_API_TARGET。
const TARGET = process.env.VITE_API_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // 端口被占时**直接失败**,不要悄悄换一个 —— 换了的话你复制到浏览器里的
    // 还是旧地址,会对着上一次残留的进程调试。
    strictPort: true,
    proxy: {
      // 走代理而不是让前端直连 8000:
      //   1. 同源,不用在后端开 CORS;
      //   2. `scripts/api_server.py` 明确说了调前端**不要**用 `--reload`
      //      (uvicorn 会在子进程里重建整个后端单例,每次都重新加载 bge-m3)。
      //      改前端只要 Vite 热更,后端那个进程一直活着。
      '/api': {
        target: TARGET,
        changeOrigin: true,
        // http-proxy 对 `text/event-stream` 默认就是流式转发,不缓冲。
        // 这里不额外加 configure 钩子 —— 那段代码平时用不上,而对 SSE 来说
        // 它一旦写错就是把"边跑边显示"退化成"转圈几十秒再一次性出结果",
        // 属于收益为零、风险非零的改动。
      },
    },
  },
})
