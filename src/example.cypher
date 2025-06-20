// ─────────────────────────── Query ───────────────────────────

MATCH (n) RETURN count(n);

MATCH (n:Concept) RETURN n.name;

MATCH ()-[r]->() RETURN DISTINCT type(r), count(*);

MATCH (e:MedicalEntity) RETURN e.name ORDER BY e.name;

// ─────────────────────────── Delete ───────────────────────────
MATCH (n) DETACH DELETE n;

// ─────────────────────────── Visualise ───────────────────────────
MATCH p = (n:MedicalEntity)-[r]->(m:Entity) RETURN p;
MATCH (n:MedicalEntity)-[r]->(m:MedicalEntity) RETURN n.name, type(r), m.name LIMIT 5;
