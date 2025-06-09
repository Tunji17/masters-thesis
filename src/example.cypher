// ─────────────────────────── Query ───────────────────────────

MATCH (n) RETURN count(n);

MATCH (n:Concept) RETURN n.name;

MATCH ()-[r]->() RETURN DISTINCT type(r), count(*);

MATCH (e:Entity) RETURN e.name ORDER BY e.name;

// ─────────────────────────── Visualise ───────────────────────────
MATCH p = (n:Entity)-[r:treated_with|associated_with]->(m:Entity) RETURN p;