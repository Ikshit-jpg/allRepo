// Maps React globals (from CDN) into ES module exports for esbuild bundling.
const ReactGlobal = window.React;
const ReactDOMGlobal = window.ReactDOM;

export const React = ReactGlobal;
export const ReactDOM = ReactDOMGlobal;
export const { useState, useEffect, useMemo, useRef } = ReactGlobal;
export default ReactGlobal;
