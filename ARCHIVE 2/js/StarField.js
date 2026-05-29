import React, { useMemo } from "./react-shim.js";

export default function StarField({ count = 90 }) {
  const stars = useMemo(
    () =>
      Array.from({ length: count }).map((_, i) => {
        const size = Math.random() * 4 + 3; // 3-7px
        const twinkle = Math.random() * 1 + 2.5; // 2.5s-3.5s
        const drift = Math.random() * 2 + 6; // 6s-8s
        const alpha = Math.random() * 0.25 + 0.55; // 0.55-0.8
        const blur = Math.random() * 6 + 8; // 8px-14px
        return {
          id: i,
          size,
          top: `${Math.random() * 100}%`,
          left: `${Math.random() * 100}%`,
          delay: `${Math.random() * 5}s`,
          twinkle,
          drift,
          alpha,
          blur,
        };
      }),
    [count]
  );

  return (
    <div className="stars-container">
      {stars.map((s) => (
        <div
          key={s.id}
          className="star"
          style={{
            width: `${s.size}px`,
            height: `${s.size}px`,
            top: s.top,
            left: s.left,
            animationDelay: s.delay,
            "--twinkle": `${s.twinkle}s`,
            "--drift": `${s.drift}s`,
            "--star-alpha": s.alpha,
            "--star-blur": `${s.blur}px`,
          }}
        />
      ))}
    </div>
  );
}
