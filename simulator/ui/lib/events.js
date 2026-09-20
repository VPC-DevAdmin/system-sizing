/* Tiny emitter for one-way notifications between tab modules, so a
 * module that only needs to TELL another something does not import
 * it and the import graph stays acyclic. Names in use:
 *   run:started, run:finished   Live, off the telemetry socket
 *   engines:changed             Engines, after every status refresh
 *   models:changed              Models / Control: the catalog or the
 *                               set of cached models changed
 *   status                      Control, every status poll
 *                               ({ active, running })
 *   disconnected                Control, when the service stops
 *                               answering */

const listeners = new Map();

export function on(name, fn) {
  if (!listeners.has(name)) listeners.set(name, new Set());
  listeners.get(name).add(fn);
  return () => listeners.get(name).delete(fn);
}

export function emit(name, payload) {
  for (const fn of listeners.get(name) ?? []) fn(payload);
}
