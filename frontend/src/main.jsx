import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './styles.css'

// 不开 StrictMode 的话开发期的双调用检查就没了(能揪出"effect 里发了两次
// 请求"这类问题)。这里的 effect 都带 abort/cleanup,经得起双调用。
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>
)
