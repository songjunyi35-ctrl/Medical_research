"""Finite workflow controllers with explicitly scoped score access."""
from collections import defaultdict
import hashlib
import math
import random
import statistics

CONTROLLERS = ('exhaustive', 'random', 'greedy', 'uct')


def views_for(workflow):
    if type(workflow) is not int or not 0 <= workflow < 128:
        raise ValueError('workflow ID must be integer 0..127')
    return (0,) + tuple(i+1 for i in range(7) if workflow & (1 << i))


def patient_groups(rows):
    ids = sorted({r['subject_id'] for r in rows},
                 key=lambda s: hashlib.sha256(('2026091131:'+s).encode()).hexdigest())
    if len(ids) != 78 or len({r['case_id'] for r in rows}) != 180:
        raise ValueError('expected exactly 180 unique scans / 78 subjects')
    groups = [ids[i:i+26] for i in range(0,78,26)]
    return groups


def scoped_rows(table, allowed_subjects):
    allowed = set(allowed_subjects)
    if not allowed:
        raise ValueError('empty scope')
    scoped = {int(k): [r for r in rows if r['subject_id'] in allowed]
              for k, rows in table.items()}
    if any({r['subject_id'] for r in rows} != allowed for rows in scoped.values()):
        raise ValueError('incomplete subject scope')
    return scoped


def metrics(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['subject_id']].append(r['dice'])
    return {'scan':statistics.fmean(r['dice'] for r in rows),
            'subject':statistics.fmean(statistics.fmean(v) for v in groups.values()),
            'point':sum(r['point_hit'] for r in rows)}


def reward(rows, workflow):
    return metrics(rows)['subject'] - .001*len(views_for(workflow))


def search_controller(name, search_rows, seed, budget=16):
    if name not in CONTROLLERS or set(search_rows) != set(range(128)):
        raise ValueError('unknown controller or incomplete workflow space')
    if not 8 <= budget <= 128:
        raise ValueError('budget must cover selector pool')
    rng = random.Random(seed)
    evaluated = {}
    query_order = []

    def query(k):
        if k not in evaluated:
            if name != 'exhaustive' and len(evaluated) >= budget:
                raise RuntimeError('unique query budget exceeded')
            evaluated[k] = reward(search_rows[k], k)
            query_order.append(k)
        return evaluated[k]

    query(0); query(127)
    simulations = 0
    if name == 'exhaustive':
        for k in range(128):
            query(k)
    elif name == 'random':
        for k in rng.sample(list(range(1,127)), budget-2):
            query(k)
    elif name == 'greedy':
        expanded = set()
        while len(evaluated) < budget:
            frontier = sorted(set(evaluated)-expanded, key=lambda k:(-evaluated[k],k))
            if not frontier:
                raise RuntimeError('connected cube frontier exhausted')
            k = frontier[0]
            expanded.add(k)
            for neighbor in sorted(k ^ (1 << bit) for bit in range(7)):
                if len(evaluated) == budget:
                    break
                query(neighbor)
    else:
        visits, totals = defaultdict(int), defaultdict(float)
        while len(evaluated) < budget and simulations < 512:
            path = [()]
            node = ()
            while len(node) < 7:
                children = (node+(0,), node+(1,))
                unseen = [c for c in children if visits[c] == 0]
                if unseen:
                    node = rng.choice(unseen)
                else:
                    scores = {c:totals[c]/visits[c] + math.sqrt(2*math.log(max(1,visits[node]))/visits[c])
                              for c in children}
                    best = max(scores.values())
                    node = rng.choice([c for c in children if scores[c] == best])
                path.append(node)
            k = sum(v << bit for bit,v in enumerate(node))
            value = query(k)
            for node in path:
                visits[node] += 1
                totals[node] += value
            simulations += 1
    ranked = sorted((k for k in evaluated if k not in (0,127)), key=lambda k:(-evaluated[k],k))
    pool = [0,127]+ranked[:6]
    return {'controller':name,'query_order':query_order,'search_rewards':evaluated,
            'selector_pool':pool,'simulations':simulations,'unique_queries':len(evaluated)}


def select_workflow(pool, selector_rows):
    if len(pool) != 8 or len(set(pool)) != 8 or not {0,127} <= set(pool):
        raise ValueError('eight candidates including both anchors required')
    if set(selector_rows) != set(pool):
        raise ValueError('selector must receive only frozen candidate pool')
    summaries = {k:metrics(rows) for k,rows in selector_rows.items()}
    full = summaries[127]
    floor = max(summaries[0]['point'], full['point'])-1
    feasible = [k for k,s in summaries.items() if s['scan']-full['scan'] >= -.003
                and s['subject']-full['subject'] >= -.003 and s['point'] >= floor]
    chosen = min(feasible,key=lambda k:(len(views_for(k)),-summaries[k]['subject'],k)) if feasible else 127
    return {'selected':chosen,'views':views_for(chosen),'fallback':not bool(feasible),
            'feasible':sorted(feasible),'selector_metrics':summaries}
