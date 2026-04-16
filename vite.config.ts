import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path';

export default defineConfig({
  base: '/chess/',
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  // TensorFlow.js requires eval() for WebGL shader compilation
  server: {
    headers: {
      'Content-Security-Policy': "script-src 'self' 'unsafe-eval'; worker-src 'self' blob:",
    },
  },
});
