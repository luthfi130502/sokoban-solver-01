// Sokoban solver: push-optimal A* per room.
//
// The levels this contest draws are a hallway with 1-4 rooms off it; a box pushed into the hallway can never
// come back, so each room is an independent push puzzle and the level's optimum is the sum of its rooms' optima.
// Each room is solved with an A* over pushes (a state is the boxes' squares plus the player's reachable region,
// named by its lowest square), guided by a box-to-goal assignment bound built from single-box push distances, and
// pruned by dead squares and frozen 2x2 blocks. The search budgets itself by expansions, never by the clock, so
// two runs of it answer byte for byte alike.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Direction letters by index: 0 up, 1 down, 2 left, 3 right; a push is the upper-case letter. */
const WALK = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  goalSet: Uint8Array;
  boxes: number[];
  player: number;
}

/** A level in XSB text: width, height, walls (short rows and the outer frame padded as walls), goals, boxes, player. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goalSet = new Uint8Array(w * h);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet[i] = 1;
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

/**
 * For each goal, the fewest pushes that bring a lone box from every square to it (the other boxes ignored), by a
 * pull search back from the goal: a box on x came from x-d only if the player could stand on x-2d. -1: never.
 */
function goalDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array[] {
  const size = wall.length;
  return goals.map((g) => {
    const dist = new Int32Array(size).fill(-1);
    dist[g] = 0;
    const queue = [g];
    for (let qi = 0; qi < queue.length; qi++) {
      const x = queue[qi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const y = x - d;
        const stand = y - d;
        if (stand < 0 || stand >= size) continue;
        if (wall[y] === 0 && wall[stand] === 0 && dist[y] < 0) {
          dist[y] = dist[x] + 1;
          queue.push(y);
        }
      }
    }
    return dist;
  });
}

const INF = 1e9;

/**
 * A lower bound on the pushes a position still needs: the cheapest assignment of distinct goals to boxes over the
 * single-box push distances (a small assignment DP for up to 5 boxes, the sum of per-box minima beyond that, which
 * is a bound too since one push moves one box). INF when some box can reach no goal at all.
 */
function bound(dists: Int32Array[], boxes: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 5) {
    let sum = 0;
    for (const b of boxes) {
      let m = INF;
      for (const d of dists) if (d[b] >= 0 && d[b] < m) m = d[b];
      if (m >= INF) return INF;
      sum += m;
    }
    return sum;
  }
  // dp[mask]: cheapest cost of parking the first popcount(mask) boxes on the goals named by mask.
  const dp = new Float64Array(1 << n).fill(INF);
  dp[0] = 0;
  for (let mask = 0; mask < dp.length; mask++) {
    const i = popcount(mask);
    if (i >= n || dp[mask] >= INF) continue;
    for (let j = 0; j < n; j++) {
      if (mask & (1 << j)) continue;
      const c = dists[j][boxes[i]];
      if (c < 0) continue;
      const next = mask | (1 << j);
      if (dp[mask] + c < dp[next]) dp[next] = dp[mask] + c;
    }
  }
  return dp[dp.length - 1];
}

function popcount(x: number): number {
  let c = 0;
  while (x) {
    x &= x - 1;
    c++;
  }
  return c;
}

interface Solved {
  moves: string;
  pushes: number;
  endPlayer: number;
}

/**
 * A push search on one board: A* (weight 1) or a weighted dive for any solution (weight > 1) over states of sorted
 * boxes plus the player's region, at most `cap` expansions. Returns the moves, or null when it gives up.
 */
function search(
  w: number,
  wall: Uint8Array,
  goals: number[],
  goalSet: Uint8Array,
  boxes0: number[],
  player0: number,
  cap: number,
  weight: number,
): Solved | null {
  const size = wall.length;
  const dists = goalDistances(w, wall, goals);
  const dead = new Uint8Array(size);
  for (let s = 0; s < size; s++) {
    dead[s] = dists.every((d) => d[s] < 0) ? 1 : 0;
  }
  if (boxes0.some((b) => dead[b] === 1 && goalSet[b] === 0)) return null;
  if (boxes0.every((b) => goalSet[b] === 1)) return { moves: '', pushes: 0, endPlayer: player0 };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  // The squares the latest flood visited, in visit order.
  const floodList: number[] = [];
  // Flood from `from` over squares free of walls and boxes; marks `seen` with a fresh stamp; returns the lowest square.
  const flood = (from: number): number => {
    stamp++;
    floodList.length = 0;
    let low = from;
    let n = 0;
    stack[n++] = from;
    seen[from] = stamp;
    floodList.push(from);
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      const up = p - w;
      const dn = p + w;
      const lf = p - 1;
      const rt = p + 1;
      if (wall[up] === 0 && occupied[up] === 0 && seen[up] !== stamp) { seen[up] = stamp; stack[n++] = up; floodList.push(up); }
      if (wall[dn] === 0 && occupied[dn] === 0 && seen[dn] !== stamp) { seen[dn] = stamp; stack[n++] = dn; floodList.push(dn); }
      if (wall[lf] === 0 && occupied[lf] === 0 && seen[lf] !== stamp) { seen[lf] = stamp; stack[n++] = lf; floodList.push(lf); }
      if (wall[rt] === 0 && occupied[rt] === 0 && seen[rt] !== stamp) { seen[rt] = stamp; stack[n++] = rt; floodList.push(rt); }
    }
    return low;
  };
  // Is the box just pushed to t frozen in a 2x2 of walls and boxes with some box off its goal?
  const frozen = (t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (a < 0 || d >= size) continue;
      const solid = (s: number): boolean => wall[s] === 1 || occupied[s] === 1;
      if (!solid(a) || !solid(b) || !solid(c) || !solid(d)) continue;
      const offGoal = (s: number): boolean => occupied[s] === 1 && goalSet[s] === 0;
      if (offGoal(a) || offGoal(b) || offGoal(c) || offGoal(d)) return true;
    }
    return false;
  };

  const keyOf = (boxes: number[], low: number): string => String.fromCharCode(low, ...boxes);

  // Node store; a node's player square is where the player stands in it (the pushed box's old square).
  const nBoxes: number[][] = [boxes0.slice()];
  const nPlayer: number[] = [player0];
  const nG: number[] = [0];
  const nParent: number[] = [-1];
  const nFrom: number[] = [-1];
  const nTo: number[] = [-1];
  const nDir: number[] = [-1];
  const best = new Map<string, number>();

  for (const b of boxes0) occupied[b] = 1;
  const low0 = flood(player0);
  for (const b of boxes0) occupied[b] = 0;
  best.set(keyOf(boxes0, low0), 0);

  // Binary heap on (f, then smaller h, then insertion order): fully deterministic.
  const heapF: number[] = [];
  const heapH: number[] = [];
  const heapN: number[] = [];
  let counter = 0;
  const heapC: number[] = [];
  const less = (i: number, j: number): boolean =>
    heapF[i] < heapF[j] ||
    (heapF[i] === heapF[j] && (heapH[i] < heapH[j] || (heapH[i] === heapH[j] && heapC[i] < heapC[j])));
  const heapPush = (f: number, h: number, node: number): void => {
    heapF.push(f);
    heapH.push(h);
    heapN.push(node);
    heapC.push(counter++);
    let i = heapF.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(i, p)) break;
      [heapF[i], heapF[p]] = [heapF[p], heapF[i]];
      [heapH[i], heapH[p]] = [heapH[p], heapH[i]];
      [heapN[i], heapN[p]] = [heapN[p], heapN[i]];
      [heapC[i], heapC[p]] = [heapC[p], heapC[i]];
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heapN[0];
    const lf = heapF.pop() as number;
    const lh = heapH.pop() as number;
    const ln = heapN.pop() as number;
    const lc = heapC.pop() as number;
    if (heapF.length > 0) {
      heapF[0] = lf;
      heapH[0] = lh;
      heapN[0] = ln;
      heapC[0] = lc;
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= heapF.length) break;
        if (c + 1 < heapF.length && less(c + 1, c)) c++;
        if (!less(c, i)) break;
        [heapF[i], heapF[c]] = [heapF[c], heapF[i]];
        [heapH[i], heapH[c]] = [heapH[c], heapH[i]];
        [heapN[i], heapN[c]] = [heapN[c], heapN[i]];
        [heapC[i], heapC[c]] = [heapC[c], heapC[i]];
        i = c;
      }
    }
    return top;
  };

  const h0 = bound(dists, boxes0);
  if (h0 >= INF) return null;
  heapPush(h0 * weight, h0, 0);
  let found = -1;
  let expansions = 0;
  while (heapF.length > 0) {
    const cur = heapPop();
    const boxes = nBoxes[cur];
    const g = nG[cur];
    // Stale heap entry? Recompute the key cheaply is not possible; track by a closed check on g via best map at push time.
    if (boxes.every((b) => goalSet[b] === 1)) {
      found = cur;
      break;
    }
    if (expansions >= cap) return null;
    expansions++;
    for (const b of boxes) occupied[b] = 1;
    flood(nPlayer[cur]);
    // Snapshot the reachable squares: the child floods below overwrite `seen` as they go.
    const reach = new Uint8Array(size);
    for (const s of floodList) reach[s] = 1;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let k = 0; k < 4; k++) {
        const d = DY[k] * w + DX[k];
        const stand = b - d;
        const dest = b + d;
        if (stand < 0 || dest < 0 || stand >= size || dest >= size) continue;
        if (reach[stand] !== 1 || wall[dest] === 1 || occupied[dest] === 1 || dead[dest] === 1) continue;
        occupied[b] = 0;
        occupied[dest] = 1;
        const isFrozen = frozen(dest);
        occupied[dest] = 0;
        occupied[b] = 1;
        if (isFrozen) continue;
        const next = boxes.slice();
        next[bi] = dest;
        next.sort((x, y) => x - y);
        const h = bound(dists, next);
        if (h >= INF) continue;
        // Region of the new state: the player stands on b now.
        for (const x of boxes) occupied[x] = 0;
        for (const x of next) occupied[x] = 1;
        const low = flood(b);
        for (const x of next) occupied[x] = 0;
        for (const x of boxes) occupied[x] = 1;
        const key = keyOf(next, low);
        const g2 = g + 1;
        const prev = best.get(key);
        if (prev !== undefined && prev <= g2) continue;
        best.set(key, g2);
        const node = nBoxes.length;
        nBoxes.push(next);
        nPlayer.push(b);
        nG.push(g2);
        nParent.push(cur);
        nFrom.push(b);
        nTo.push(dest);
        nDir.push(k);
        heapPush(g2 + weight * h, h, node);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;
  // Rebuild the push list, then the moves: walk to each push's stand square, then push.
  const chain: number[] = [];
  for (let s = found; s > 0; s = nParent[s]) chain.push(s);
  chain.reverse();
  let moves = '';
  let player = player0;
  const cur = boxes0.slice();
  for (const b of cur) occupied[b] = 1;
  for (const s of chain) {
    const from = nFrom[s];
    const to = nTo[s];
    const k = nDir[s];
    const d = DY[k] * w + DX[k];
    moves += walk(w, wall, occupied, player, from - d);
    moves += PUSH[k];
    occupied[from] = 0;
    occupied[to] = 1;
    cur[cur.indexOf(from)] = to;
    player = from;
  }
  for (const b of cur) occupied[b] = 0;
  return { moves, pushes: chain.length, endPlayer: player };
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower letters (first direction wins ties). */
function walk(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Int32Array(size).fill(-1);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k] * w + DX[k];
      if (q < 0 || q >= size || wall[q] === 1 || occupied[q] === 1 || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = k;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = WALK[how[p]] + s;
  return s;
}

interface Room {
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: no box there can ever reach a goal)
 * out of the floor; each remaining part that holds a box is a room. Null when the level is not such a set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const dists = goalDistances(w, wall, goals);
  const live = new Uint8Array(wall.length);
  for (let s = 0; s < wall.length; s++) live[s] = dists.some((d) => d[s] >= 0) ? 1 : 0;
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (wall[i] === 0 && live[i] === 0) cut[i] = 1;
  if (boxes.some((b) => cut[b] === 1)) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] === 1 || cut[s] === 1 || part[s] >= 0) continue;
    const id = rooms.length;
    const squares = [s];
    part[s] = id;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i] + DY[k] * w + DX[k];
        if (q >= 0 && q < wall.length && wall[q] === 0 && cut[q] === 0 && part[q] < 0) {
          part[q] = id;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  const withBoxes = rooms.filter((r) => r.boxes.length > 0);
  for (const r of withBoxes) {
    r.boxes.sort((a, b) => a - b);
    r.goals.sort((a, b) => a - b);
  }
  return withBoxes;
}

/** Solve one room on a board where the other rooms' boxes (or their goals, once done) stand as walls. */
function solveRoom(level: Level, rooms: Room[], room: Room, player: number): Solved | null {
  const wall = level.wall.slice();
  for (const other of rooms) {
    if (other === room) continue;
    for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
  }
  const goalSet = new Uint8Array(level.wall.length);
  for (const g of room.goals) goalSet[g] = 1;
  return (
    search(level.w, wall, room.goals, goalSet, room.boxes, player, 1_200_000, 1) ??
    search(level.w, wall, room.goals, goalSet, room.boxes, player, 1_200_000, 3)
  );
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.length === 0 || level.boxes.every((b) => level.goalSet[b] === 1)) return '';
  const rooms = roomsOf(level);
  if (rooms !== null && rooms.length > 0) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const solved = solveRoom(level, rooms, room, player);
      if (solved === null) {
        ok = false;
        break;
      }
      moves += solved.moves;
      player = solved.endPlayer;
      room.done = true;
    }
    if (ok) return moves;
  }
  // Fallback: the whole level at once.
  const whole =
    search(level.w, level.wall, level.goals, level.goalSet, level.boxes, level.player, 1_500_000, 1) ??
    search(level.w, level.wall, level.goals, level.goalSet, level.boxes, level.player, 1_500_000, 3);
  return whole === null ? '' : whole.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
