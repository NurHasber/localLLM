(function registerLocalPhotoAiSW() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.getRegistrations().then((regs) => {
    regs.forEach((r) => {
      const src = (r.active && r.active.scriptURL) || (r.installing && r.installing.scriptURL) || "";
      if (src.includes("/static/sw.js")) r.unregister();
    });
    return navigator.serviceWorker.register("/sw.js", { scope: "/" });
  }).catch(() => {});
})();
