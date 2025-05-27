// ─────────────────────────── Query ───────────────────────────

MATCH (n:Concept) RETURN n.name;
MATCH ()-[r]->() RETURN DISTINCT type(r), count(*);

// ─────────────────────────── Visualise ───────────────────────────
MATCH p = (c1:Concept)-[r:IS_PART_OF|TREATS]->(c2:Concept)
RETURN p;