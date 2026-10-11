const START_URL = "localphotoai://start";
const PANEL_URL = "http://127.0.0.1:7860/";
const btn = document.getElementById("btnStart");
const status = document.getElementById("status");
let waiting = false;

function invokeLocalProtocol(url) {
  const iframe = document.createElement("iframe");
  iframe.style.cssText = "display:none;width:0;height:0;border:0";
  iframe.src = url;
  document.body.appendChild(iframe);
  setTimeout(() => iframe.remove(), 2500);
}

async function panelIsUp() {
  try {
    const res = await fetch("/api/status?_=" + Date.now(), { cache: "no-store" });
    if (!res.ok) return false;
    const data = await res.json().catch(() => null);
    return !!(data && !data.offline);
  } catch {
    return false;
  }
}

async function waitAndReload() {
  if (waiting) return;
  waiting = true;
  btn.disabled = true;
  btn.textContent = "Запускаю…";
  for (let i = 1; i <= 40; i++) {
    status.textContent = "Жду запуск сервиса… (" + i + "/40)";
    if (i === 1 || i % 4 === 0) invokeLocalProtocol(START_URL);
    if (await panelIsUp()) {
      status.textContent = "Готово, обновляю вкладку…";
      window.location.replace(PANEL_URL + "?t=" + Date.now());
      return;
    }
    await new Promise((r) => setTimeout(r, 700));
  }
  waiting = false;
  status.textContent = "Не ответил. Нажмите ещё раз или откройте start-tray.bat / launcher.html";
  btn.disabled = false;
  btn.textContent = "Запустить Photone";
}

btn.addEventListener("click", () => {
  status.textContent = "Запускаю локальный сервис…";
  invokeLocalProtocol(START_URL);
  waitAndReload();
});

(async () => {
  if (await panelIsUp()) {
    window.location.replace(PANEL_URL + "?t=" + Date.now());
  }
})();
