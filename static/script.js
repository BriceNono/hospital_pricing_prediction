/* MedPrice AI — front-end logic
   Handles: model health check, form submission, /predict call,
   and the gauge / readout animation for the result panel. */

(function () {
  "use strict";

  // Observed range of Treatment_Cost_USD in the training dataset, used only
  // to scale the gauge visually (min/max are not re-derived server-side).
  const COST_GAUGE_MIN = 600;
  const COST_GAUGE_MAX = 5300;
  const GAUGE_ARC_LENGTH = 267; // matches stroke-dasharray in style.css

  const form = document.getElementById("predictForm");
  const submitBtn = document.getElementById("submitBtn");
  const formError = document.getElementById("formError");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const resultState = document.getElementById("resultState");

  const gaugeFill = document.getElementById("gaugeFill");
  const amountReadout = document.getElementById("amountReadout");
  const resultRange = document.getElementById("resultRange");
  const resultMeta = document.getElementById("resultMeta");

  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");

  function showState(which) {
    emptyState.hidden = which !== "empty";
    loadingState.hidden = which !== "loading";
    resultState.hidden = which !== "result";
  }

  function setError(message) {
    if (!message) {
      formError.textContent = "";
      formError.classList.remove("visible");
      return;
    }
    formError.textContent = message;
    formError.classList.add("visible");
  }

  function animateCount(el, target) {
    const duration = 900;
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const value = Math.round(target * eased);
      el.textContent = value.toLocaleString("en-US");
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function updateGauge(cost) {
    const clamped = Math.min(Math.max(cost, COST_GAUGE_MIN), COST_GAUGE_MAX);
    const pct = (clamped - COST_GAUGE_MIN) / (COST_GAUGE_MAX - COST_GAUGE_MIN);
    // Reset first so the transition always plays from empty
    gaugeFill.style.transition = "none";
    gaugeFill.style.strokeDashoffset = String(GAUGE_ARC_LENGTH);
    // Force reflow, then animate to the target offset
    void gaugeFill.getBoundingClientRect();
    gaugeFill.style.transition = "";
    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = String(GAUGE_ARC_LENGTH * (1 - pct));
    });
  }

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      if (res.ok && data.status === "ok") {
        statusDot.classList.add("ok");
        statusText.textContent = "Model ready";
      } else {
        statusDot.classList.add("error");
        statusText.textContent = "Model unavailable";
      }
    } catch (err) {
      statusDot.classList.add("error");
      statusText.textContent = "Server unreachable";
    }
  }

  function collectPayload() {
    const fd = new FormData(form);
    return {
      Age: fd.get("Age"),
      Gender: fd.get("Gender"),
      Diagnosis: fd.get("Diagnosis"),
      Treatment_Type: fd.get("Treatment_Type"),
      Length_of_Stay: fd.get("Length_of_Stay"),
      BMI: fd.get("BMI"),
      Smoker: fd.get("Smoker"),
      Resource_Utilization: fd.get("Resource_Utilization"),
    };
  }

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    setError(null);

    submitBtn.disabled = true;
    showState("loading");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
      });
      const data = await response.json();

      if (!response.ok) {
        setError(data.error || "Something went wrong while predicting the cost.");
        showState("empty");
        return;
      }

      const cost = data.predicted_cost_usd;
      const low = Math.round(cost * 0.9);
      const high = Math.round(cost * 1.1);

      showState("result");
      updateGauge(cost);
      animateCount(amountReadout, Math.round(cost));
      resultRange.textContent = `Likely range: $${low.toLocaleString("en-US")} – $${high.toLocaleString("en-US")}`;

      resultMeta.innerHTML = "";
      const input = data.input || {};
      const chips = [
        input.Diagnosis,
        input.Treatment_Type,
        `${input.Length_of_Stay} day stay`,
        `${input.Resource_Utilization} utilization`,
      ].filter(Boolean);
      chips.forEach((label) => {
        const span = document.createElement("span");
        span.textContent = label;
        resultMeta.appendChild(span);
      });
    } catch (err) {
      setError("Could not reach the prediction service. Please try again.");
      showState("empty");
    } finally {
      submitBtn.disabled = false;
    }
  });

  checkHealth();
})();
