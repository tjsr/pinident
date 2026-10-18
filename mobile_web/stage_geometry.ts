export interface FramePlacement {
  fit: number;
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Match CSS object-fit: contain inside the preview stage. */
export function containFrame(stageWidth: number, stageHeight: number,
                             frameWidth: number, frameHeight: number): FramePlacement | null {
  if (![stageWidth, stageHeight, frameWidth, frameHeight].every(value =>
    Number.isFinite(value) && value > 0)) return null;
  const fit = Math.min(stageWidth / frameWidth, stageHeight / frameHeight);
  const width = frameWidth * fit, height = frameHeight * fit;
  return { fit, width, height, left: (stageWidth - width) / 2, top: (stageHeight - height) / 2 };
}

export function framePoint(placement: FramePlacement, stagePoint: [number, number],
                           frameWidth: number, frameHeight: number): [number, number] | null {
  const [x, y] = stagePoint;
  if (x < placement.left || y < placement.top ||
      x > placement.left + placement.width || y > placement.top + placement.height) return null;
  return [Math.min(frameWidth, Math.max(0, (x - placement.left) / placement.fit)),
    Math.min(frameHeight, Math.max(0, (y - placement.top) / placement.fit))];
}
