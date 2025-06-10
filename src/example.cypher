// ─────────────────────────── Query ───────────────────────────

MATCH (n) RETURN count(n);

MATCH (n:Concept) RETURN n.name;

MATCH ()-[r]->() RETURN DISTINCT type(r), count(*);

MATCH (e:Entity) RETURN e.name ORDER BY e.name;

// ─────────────────────────── Delete ───────────────────────────
MATCH (n) DETACH DELETE n;

// ─────────────────────────── Visualise ───────────────────────────
MATCH p = (n:Entity)-[r]->(m:Entity) RETURN p;