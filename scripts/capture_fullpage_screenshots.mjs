import { spawn } from 'child_process';
import http from 'http';
import fs from 'fs';
import path from 'path';
import os from 'os';

const EDGE_PATH = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const PORT = 9223;
const STREAMLIT_URL = "http://localhost:8502";

const FIGURES_DIR = path.resolve('figures');
const STREAMLIT_DIR = path.resolve('streamlit');
const TEMP_USER_DATA = path.join(os.tmpdir(), `edge_cdp_profile_${Date.now()}`);

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function fetchJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    }).on('error', reject);
  });
}

async function waitForHttp(url, maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const data = await fetchJson(url);
      return data;
    } catch {
      await sleep(300);
    }
  }
  throw new Error(`Timeout waiting for ${url}`);
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.id = 1;
    this.callbacks = new Map();
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => resolve();
      this.ws.onerror = err => reject(err);
      this.ws.onmessage = msg => {
        const data = JSON.parse(msg.data);
        if (data.id && this.callbacks.has(data.id)) {
          const { resolve, reject } = this.callbacks.get(data.id);
          this.callbacks.delete(data.id);
          if (data.error) {
            reject(new Error(data.error.message || JSON.stringify(data.error)));
          } else {
            resolve(data.result);
          }
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true
    });
    return res.result?.value;
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

async function run() {
  console.log("Launching Edge headless browser...");
  const edgeProcess = spawn(EDGE_PATH, [
    '--headless=new',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${TEMP_USER_DATA}`,
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--window-size=1280,900',
    STREAMLIT_URL
  ], {
    detached: false,
    stdio: 'ignore'
  });

  try {
    console.log("Waiting for browser CDP endpoint...");
    await waitForHttp(`http://127.0.0.1:${PORT}/json/version`);

    const targets = await fetchJson(`http://127.0.0.1:${PORT}/json/list`);
    const pageTarget = targets.find(t => t.type === 'page');
    if (!pageTarget || !pageTarget.webSocketDebuggerUrl) {
      throw new Error("No page target found with WebSocket debugger URL");
    }

    console.log("Connecting to target via WebSocket...");
    const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await client.connect();

    await client.send('Page.enable');
    await client.send('DOM.enable');
    await client.send('Runtime.enable');

    console.log("Waiting for Streamlit app to fully load...");
    for (let i = 0; i < 40; i++) {
      const ready = await client.evaluate(`
        Boolean(document.querySelector('textarea') && document.querySelector('button'))
      `);
      if (ready) break;
      await sleep(500);
    }
    await sleep(2000); // Wait for Streamlit animations/hydration to settle

    // Helper to capture and save full page screenshot
    async function captureAndSave(filenameBase, description) {
      console.log(`Capturing: ${description}...`);
      const screenshot = await client.send('Page.captureScreenshot', {
        format: 'png',
        captureBeyondViewport: true
      });
      const buffer = Buffer.from(screenshot.data, 'base64');
      
      const fileFigures = path.join(FIGURES_DIR, `${filenameBase}.png`);
      const fileStreamlit = path.join(STREAMLIT_DIR, `${filenameBase}.png`);
      
      fs.writeFileSync(fileFigures, buffer);
      fs.writeFileSync(fileStreamlit, buffer);
      console.log(`Saved: ${fileFigures} and ${fileStreamlit}`);
    }

    // Helper to reload clean page
    async function reloadCleanPage() {
      await client.send('Page.navigate', { url: STREAMLIT_URL });
      for (let i = 0; i < 40; i++) {
        const ready = await client.evaluate(`
          Boolean(document.querySelector('textarea') && document.querySelector('button'))
        `);
        if (ready) break;
        await sleep(500);
      }
      await sleep(1500);
    }

    // 1. Initial State (Blank review input)
    await reloadCleanPage();
    await captureAndSave('screenshot_01_initial_page', 'Initial Streamlit state');

    // 2. Positive Review Test
    console.log("Submitting positive review test...");
    await reloadCleanPage();
    await client.evaluate(`
      (() => {
        const ta = document.querySelector('textarea');
        ta.focus();
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        nativeSetter.call(ta, "The hotel room was very clean, spacious, and the staff was exceptionally friendly and helpful! Outstanding breakfast buffet.");
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        ta.dispatchEvent(new Event('change', { bubbles: true }));
        ta.dispatchEvent(new Event('blur', { bubbles: true }));
        ta.blur();
      })()
    `);
    await sleep(1500);

    await client.evaluate(`
      (() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const btn = buttons.find(b => b.innerText.includes('Predict'));
        if (btn) btn.click();
      })()
    `);

    // Wait for prediction result
    for (let i = 0; i < 40; i++) {
      const hasPrediction = await client.evaluate(`
        document.body.innerText.includes('Positive Review')
      `);
      if (hasPrediction) break;
      await sleep(400);
    }
    await sleep(2000);
    await captureAndSave('screenshot_02_positive_prediction', 'Positive review prediction');

    // 3. Negative Review Test
    console.log("Submitting negative review test...");
    await reloadCleanPage();
    await client.evaluate(`
      (() => {
        const ta = document.querySelector('textarea');
        ta.focus();
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        nativeSetter.call(ta, "Services were very poor. Not getting response from the reception most of the time. Two times I got poor food and dirty room.");
        ta.dispatchEvent(new Event('input', { bubbles: true }));
        ta.dispatchEvent(new Event('change', { bubbles: true }));
        ta.dispatchEvent(new Event('blur', { bubbles: true }));
        ta.blur();
      })()
    `);
    await sleep(1500);

    await client.evaluate(`
      (() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const btn = buttons.find(b => b.innerText.includes('Predict'));
        if (btn) btn.click();
      })()
    `);

    // Wait for prediction result
    for (let i = 0; i < 40; i++) {
      const hasPrediction = await client.evaluate(`
        document.body.innerText.includes('Negative Review')
      `);
      if (hasPrediction) break;
      await sleep(400);
    }
    await sleep(2000);
    await captureAndSave('screenshot_03_negative_prediction', 'Negative review prediction');

    client.close();
    console.log("All screenshots successfully captured!");
  } finally {
    try {
      edgeProcess.kill();
    } catch {}
    // Clean up temporary profile
    setTimeout(() => {
      try {
        fs.rmSync(TEMP_USER_DATA, { recursive: true, force: true });
      } catch {}
    }, 1000);
  }
}

run().catch(err => {
  console.error("Error capturing screenshots:", err);
  process.exit(1);
});
