/** Finish only after the annotation is durable and the source has not changed. */
export async function completeScan(session: string, currentSession: () => string,
                                   save: () => Promise<boolean>, reset: () => void): Promise<boolean> {
  if (!(await save()) || currentSession() !== session) return false;
  reset();
  return true;
}

/** Source choices give way to scan actions until the current source is closed. */
export function scanControls(hasSource: boolean, hasFrame: boolean, still: boolean,
                             imageSource: boolean) {
  const running = hasSource && !still && !imageSource;
  return {
    running,
    sourceChoices: !hasSource,
    stop: hasSource && !running,
    browse: !running,
    freeze: running,
    resume: still && !imageSource,
    finish: hasFrame,
    stillActions: still && hasFrame,
  };
}
