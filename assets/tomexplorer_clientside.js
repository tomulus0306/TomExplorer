(function () {
  const SUBSCRIPT_DIGITS = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
  };

  const html = (type, props) => ({
    namespace: "dash_html_components",
    type,
    props,
  });

  function displayFormula(gas) {
    return String(gas).replace(/[0-9]/g, (digit) => SUBSCRIPT_DIGITS[digit] || digit);
  }

  function formatConcentration(molarFraction) {
    if (molarFraction >= 1.0e-2) {
      return `${(molarFraction * 100).toPrecision(3)} %`;
    }
    if (molarFraction >= 1.0e-6) {
      return `${(molarFraction * 1.0e6).toPrecision(3)} ppm`;
    }
    return `${(molarFraction * 1.0e9).toPrecision(3)} ppb`;
  }

  function defaultPanel() {
    return html("Div", {
      className: "hover-card",
      children: [
        html("H4", { className: "hover-title", children: "Hover-Details" }),
        html("P", {
          className: "section-copy",
          children: "Wellenlänge, Wellenzahl sowie α und σ der Komponenten erscheinen hier beim Hover über dem Spektrum.",
        }),
      ],
    });
  }

  function nearestIndex(values, target) {
    let bestIndex = 0;
    let bestDistance = Infinity;
    for (let index = 0; index < values.length; index += 1) {
      const distance = Math.abs(Number(values[index]) - target);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = index;
      }
    }
    return bestIndex;
  }

  function normalizeCustomdata(customdata) {
    if (Array.isArray(customdata)) {
      return customdata;
    }
    if (customdata && typeof customdata === "object") {
      return Object.keys(customdata)
        .sort((left, right) => Number(left) - Number(right))
        .map((key) => customdata[key]);
    }
    return [];
  }

  function currentGraphHoverData(graphId) {
    const graphHost = document.getElementById(graphId);
    const graph = graphHost ? graphHost.querySelector(".js-plotly-plot") : null;
    if (!graph || !Array.isArray(graph._hoverdata) || graph._hoverdata.length === 0) {
      return null;
    }
    const point = graph._hoverdata.find((entry) => normalizeCustomdata(entry && entry.customdata).length > 0) || graph._hoverdata[0];
    if (!point) {
      return null;
    }
    return {
      points: [
        {
          x: point.x,
          customdata: normalizeCustomdata(point.customdata),
        },
      ],
    };
  }

  function getDashComponentProps(componentId) {
    const store = window.dash_stores && window.dash_stores[0];
    if (!store || typeof store.getState !== "function") {
      return null;
    }

    const state = store.getState();
    const stack = [state.layout];
    while (stack.length > 0) {
      const current = stack.pop();
      if (!current || typeof current !== "object") {
        continue;
      }
      if (current.props && current.props.id === componentId) {
        return current.props;
      }
      for (const value of Object.values(current)) {
        if (value && typeof value === "object") {
          stack.push(value);
        }
      }
    }
    return null;
  }

  function setPanelChildren(panelId, children) {
    if (!window.dash_clientside || typeof window.dash_clientside.set_props !== "function") {
      return;
    }
    window.dash_clientside.set_props(panelId, { children });
  }

  function hoverSignature(hoverData, serializedResult) {
    if (!serializedResult || !hoverData || !hoverData.points || hoverData.points.length === 0) {
      return "default";
    }
    const point = hoverData.points[0] || {};
    const pointCustomdata = normalizeCustomdata(point.customdata);
    return JSON.stringify([
      pointCustomdata[0] ?? null,
      pointCustomdata[1] ?? null,
      serializedResult.wavelength_um ? serializedResult.wavelength_um.length : 0,
      Object.keys(serializedResult.components || {}).length,
    ]);
  }

  function syncHoverPanel(graphId, panelId, storeId, selectResult) {
    const graphHost = document.getElementById(graphId);
    const graph = graphHost ? graphHost.querySelector(".js-plotly-plot") : null;
    if (!graph) {
      return;
    }

    const props = getDashComponentProps(storeId);
    const serializedResult = selectResult(props ? props.data : null);
    const hoverData = currentGraphHoverData(graphId);
    const signatureKey = `__tomExplorerHoverSignature_${panelId}`;
    const signature = hoverSignature(hoverData, serializedResult);
    if (graph[signatureKey] === signature) {
      return;
    }
    graph[signatureKey] = signature;
    setPanelChildren(panelId, buildHoverChildren(hoverData, serializedResult));
  }

  function ensureHoverPolling() {
    syncHoverPanel("manual-graph", "manual-hover-panel", "manual-spectrum-store", (data) => data || null);
    syncHoverPanel("search-graph", "search-hover-panel", "search-store", (data) => (data && data.spectrum ? data.spectrum : null));
  }

  function buildHoverChildren(hoverData, serializedResult) {
    if (!serializedResult || !hoverData || !hoverData.points || hoverData.points.length === 0) {
      return defaultPanel();
    }

    const point = hoverData.points[0] || {};
    const pointCustomdata = normalizeCustomdata(point.customdata);
    const hoveredWavelengthUm = pointCustomdata.length > 0 ? Number(pointCustomdata[0]) / 1000.0 : Number.NaN;
    const hoveredWavenumber = pointCustomdata.length > 1 ? Number(pointCustomdata[1]) : Number.NaN;
    const wavelengths = serializedResult.wavelength_um || [];
    const wavenumbers = serializedResult.wavenumber_cm1 || [];
    if (!wavelengths.length) {
      return defaultPanel();
    }

    let sampleIndex = 0;
    if (Number.isFinite(hoveredWavelengthUm)) {
      sampleIndex = nearestIndex(wavelengths, hoveredWavelengthUm);
    } else if (Number.isFinite(hoveredWavenumber) && wavenumbers.length) {
      sampleIndex = nearestIndex(wavenumbers, hoveredWavenumber);
    } else {
      sampleIndex = nearestIndex(wavelengths, Number(point.x));
    }
    const components = serializedResult.components || {};

    const rows = Object.keys(components).map((gas) => {
      const component = components[gas];
      const sigma = Number(component.sigma_cm2_per_molecule[sampleIndex]);
      const alpha = Number(component.alpha_per_cm[sampleIndex]);
      const concentration = Number(component.concentration);
      return html("Div", {
        className: "hover-row",
        children: [
          html("Span", {
            className: "hover-gas",
            style: { color: component.color || "#1f2937" },
            children: displayFormula(gas),
          }),
          html("Span", { children: `σ [cm²/Molekül] ${sigma.toExponential(3)}` }),
          html("Span", { children: `α [1/cm] ${alpha.toExponential(3)}` }),
          html("Span", { children: formatConcentration(concentration) }),
        ],
      });
    });

    return html("Div", {
      className: "hover-card",
      children: [
        html("H4", { className: "hover-title", children: "Hover-Details" }),
        html("Div", {
          className: "hover-metrics",
          children: [
            html("Div", {
              children: [
                html("Span", { children: "λ" }),
                html("Strong", { children: `${(Number(serializedResult.wavelength_um[sampleIndex]) * 1000).toFixed(3)} nm` }),
              ],
            }),
            html("Div", {
              children: [
                html("Span", { children: "ν" }),
                html("Strong", { children: `${Number(serializedResult.wavenumber_cm1[sampleIndex]).toFixed(3)} cm⁻¹` }),
              ],
            }),
            html("Div", {
              children: [
                html("Span", { children: "Σ α [1/cm]" }),
                html("Strong", { children: `${Number(serializedResult.total_alpha_per_cm[sampleIndex]).toExponential(3)} 1/cm` }),
              ],
            }),
            html("Div", {
              children: [
                html("Span", { children: "Σ σ [cm²/Molekül]" }),
                html("Strong", { children: `${Number(serializedResult.total_sigma_cm2_per_molecule[sampleIndex]).toExponential(3)} cm²/Molekül` }),
              ],
            }),
          ],
        }),
        html("Div", { className: "hover-table", children: rows }),
      ],
    });
  }

  window.dash_clientside = Object.assign({}, window.dash_clientside, {
    tomexplorer: {
      manual_hover_children: function (hoverData, serializedResult, currentChildren) {
        const liveHoverData = currentGraphHoverData("manual-graph");
        return buildHoverChildren(liveHoverData || hoverData, serializedResult);
      },
      search_hover_children: function (hoverData, store, currentChildren) {
        const liveHoverData = currentGraphHoverData("search-graph");
        return buildHoverChildren(liveHoverData || hoverData, store && store.spectrum ? store.spectrum : null);
      },
    },
  });

  ensureHoverPolling();
  window.setInterval(ensureHoverPolling, 150);
})();