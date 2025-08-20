/**
 * Utility functions to transform relationship data into graph format for React Flow
 */

/**
 * Transform relationships array into nodes and edges for graph visualization
 * @param {Array} relationships - Array of relationship objects with entity1, relation, entity2
 * @param {Array} entities - Array of entity objects (optional, for additional metadata)
 * @returns {Object} Object with nodes and edges arrays
 */
export function buildGraphData(relationships, entities = []) {
  if (!relationships || relationships.length === 0) {
    return { nodes: [], edges: [] };
  }

  // Extract unique entities from relationships
  const entitySet = new Set();
  const relationshipCounts = new Map();

  // Count entity occurrences for sizing
  relationships.forEach(rel => {
    entitySet.add(rel.entity1);
    entitySet.add(rel.entity2);
    
    relationshipCounts.set(rel.entity1, (relationshipCounts.get(rel.entity1) || 0) + 1);
    relationshipCounts.set(rel.entity2, (relationshipCounts.get(rel.entity2) || 0) + 1);
  });

  const uniqueEntities = Array.from(entitySet);
  
  // Create entity metadata map for additional information
  const entityMetadata = new Map();
  entities.forEach(entity => {
    entityMetadata.set(entity.text, entity);
  });

  // Generate nodes with initial random positioning (will be optimized by force-directed layout)
  const nodes = uniqueEntities.map((entityName, index) => {
    const connectionCount = relationshipCounts.get(entityName) || 1;
    const entityData = entityMetadata.get(entityName);
    
    // Initial random position - will be optimized by force-directed algorithm
    const x = 200 + Math.random() * 400; // Random position in 200-600 range
    const y = 200 + Math.random() * 400; // Random position in 200-600 range

    return {
      id: `entity-${index}`,
      type: 'default',
      position: { x, y },
      data: {
        label: entityName,
        connectionCount,
        cui: entityData?.cui,
        canonicalName: entityData?.canonical_name,
        semanticTypes: entityData?.semantic_types,
        linkingScore: entityData?.linking_score
      },
      style: {
        background: getNodeColor(connectionCount),
        color: '#333',
        border: '2px solid #222',
        width: Math.max(120, Math.min(200, entityName.length * 8 + 40)),
        height: 'auto',
        padding: '10px',
        borderRadius: '8px',
        fontSize: '12px',
        fontWeight: '500',
        textAlign: 'center'
      }
    };
  });

  // Create entity to node ID mapping
  const entityToNodeId = new Map();
  uniqueEntities.forEach((entityName, index) => {
    entityToNodeId.set(entityName, `entity-${index}`);
  });

  // Generate edges from relationships
  const edges = relationships.map((rel, index) => {
    const sourceId = entityToNodeId.get(rel.entity1);
    const targetId = entityToNodeId.get(rel.entity2);
    
    return {
      id: `edge-${index}`,
      source: sourceId,
      target: targetId,
      type: 'smoothstep',
      animated: false,
      label: rel.relation,
      style: {
        stroke: getEdgeColor(rel.relation),
        strokeWidth: 2,
      },
      labelStyle: {
        fill: '#333',
        fontWeight: '500',
        fontSize: '10px',
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        padding: '2px 4px',
        borderRadius: '3px'
      },
      markerEnd: {
        type: 'arrowclosed',
        color: getEdgeColor(rel.relation),
      },
    };
  });

  return { nodes, edges };
}

/**
 * Get node color based on connection count
 * @param {number} connectionCount 
 * @returns {string} CSS color value
 */
function getNodeColor(connectionCount) {
  if (connectionCount >= 5) return '#ef4444'; // red for highly connected
  if (connectionCount >= 3) return '#f97316'; // orange for moderately connected
  if (connectionCount >= 2) return '#eab308'; // yellow for somewhat connected
  return '#3b82f6'; // blue for minimally connected
}

/**
 * Get edge color based on relationship type
 * @param {string} relationType 
 * @returns {string} CSS color value
 */
function getEdgeColor(relationType) {
  const relationColors = {
    // Medical relationships
    'HAS_CONDITION': '#dc2626',
    'DIAGNOSED_WITH': '#b91c1c',
    'TREATED_WITH': '#059669',
    'CAUSES': '#7c2d12',
    'LOCATED_IN': '#7c3aed',
    'ASSOCIATED_WITH': '#1d4ed8',
    'INDICATES': '#be123c',
    'AFFECTS': '#ea580c',
    'RESULTS_IN': '#9333ea',
    'PREVENTS': '#16a34a',
    // Default colors for unknown relationships
    'default': '#6b7280'
  };

  return relationColors[relationType] || relationColors['default'];
}

/**
 * Advanced force-directed layout algorithm for natural graph positioning
 * @param {Array} nodes 
 * @param {Array} edges 
 * @returns {Array} Updated nodes with optimized positions
 */
export function optimizeLayout(nodes, edges) {
  if (nodes.length === 0) return nodes;
  
  const updatedNodes = [...nodes];
  const iterations = 150; // More iterations for better convergence
  const k = 120; // Optimal edge length
  const area = 800 * 600; // Canvas area
  const gravity = 0.1; // Center attraction force
  const centerX = 400;
  const centerY = 300;
  
  // Create adjacency map for faster edge lookup
  const adjacencyMap = new Map();
  updatedNodes.forEach(node => {
    adjacencyMap.set(node.id, new Set());
  });
  
  edges.forEach(edge => {
    adjacencyMap.get(edge.source)?.add(edge.target);
    adjacencyMap.get(edge.target)?.add(edge.source);
  });
  
  for (let iter = 0; iter < iterations; iter++) {
    const forces = new Map();
    const temperature = 1 - (iter / iterations); // Cooling schedule
    
    // Initialize forces
    updatedNodes.forEach(node => {
      forces.set(node.id, { fx: 0, fy: 0 });
    });
    
    // 1. Repulsive forces between all pairs of nodes (Coulomb's law)
    for (let i = 0; i < updatedNodes.length; i++) {
      for (let j = i + 1; j < updatedNodes.length; j++) {
        const nodeA = updatedNodes[i];
        const nodeB = updatedNodes[j];
        
        const dx = nodeA.position.x - nodeB.position.x || 0.01; // Avoid division by zero
        const dy = nodeA.position.y - nodeB.position.y || 0.01;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance > 0) {
          // Stronger repulsion for closer nodes
          const repulsiveForce = (k * k) / distance;
          const fx = (dx / distance) * repulsiveForce;
          const fy = (dy / distance) * repulsiveForce;
          
          forces.get(nodeA.id).fx += fx;
          forces.get(nodeA.id).fy += fy;
          forces.get(nodeB.id).fx -= fx;
          forces.get(nodeB.id).fy -= fy;
        }
      }
    }
    
    // 2. Attractive forces for connected nodes (Hooke's law)
    edges.forEach(edge => {
      const sourceNode = updatedNodes.find(n => n.id === edge.source);
      const targetNode = updatedNodes.find(n => n.id === edge.target);
      
      if (sourceNode && targetNode) {
        const dx = targetNode.position.x - sourceNode.position.x;
        const dy = targetNode.position.y - sourceNode.position.y;
        const distance = Math.sqrt(dx * dx + dy * dy) || 0.01;
        
        // Spring force proportional to distance from optimal length
        const springForce = Math.log(distance / k) * k * 0.5;
        const fx = (dx / distance) * springForce;
        const fy = (dy / distance) * springForce;
        
        forces.get(sourceNode.id).fx += fx;
        forces.get(sourceNode.id).fy += fy;
        forces.get(targetNode.id).fx -= fx;
        forces.get(targetNode.id).fy -= fy;
      }
    });
    
    // 3. Gravity force to keep nodes centered
    updatedNodes.forEach(node => {
      const dx = centerX - node.position.x;
      const dy = centerY - node.position.y;
      const distance = Math.sqrt(dx * dx + dy * dy) || 1;
      
      const gravityForce = gravity * distance * 0.01;
      forces.get(node.id).fx += (dx / distance) * gravityForce;
      forces.get(node.id).fy += (dy / distance) * gravityForce;
    });
    
    // 4. Apply forces with adaptive damping and movement limits
    const maxDisplacement = k * temperature; // Decrease movement over time
    
    updatedNodes.forEach(node => {
      const force = forces.get(node.id);
      const displacement = Math.sqrt(force.fx * force.fx + force.fy * force.fy);
      
      if (displacement > 0) {
        // Limit displacement to prevent oscillation
        const limitedDisplacement = Math.min(maxDisplacement, displacement);
        const scale = limitedDisplacement / displacement;
        
        node.position.x += force.fx * scale * 0.1;
        node.position.y += force.fy * scale * 0.1;
        
        // Keep nodes within reasonable bounds
        node.position.x = Math.max(50, Math.min(750, node.position.x));
        node.position.y = Math.max(50, Math.min(550, node.position.y));
      }
    });
  }
  
  return updatedNodes;
}

/**
 * Get relationship statistics for displaying in UI
 * @param {Array} relationships 
 * @returns {Object} Statistics object
 */
export function getGraphStats(relationships) {
  if (!relationships || relationships.length === 0) {
    return {
      totalRelationships: 0,
      uniqueEntities: 0,
      relationshipTypes: 0,
      mostConnectedEntity: null,
      relationshipTypeDistribution: {}
    };
  }

  const entities = new Set();
  const relationTypes = new Set();
  const entityConnections = new Map();
  const relationTypeDistribution = {};

  relationships.forEach(rel => {
    entities.add(rel.entity1);
    entities.add(rel.entity2);
    relationTypes.add(rel.relation);
    
    entityConnections.set(rel.entity1, (entityConnections.get(rel.entity1) || 0) + 1);
    entityConnections.set(rel.entity2, (entityConnections.get(rel.entity2) || 0) + 1);
    
    relationTypeDistribution[rel.relation] = (relationTypeDistribution[rel.relation] || 0) + 1;
  });

  // Find most connected entity
  let mostConnectedEntity = null;
  let maxConnections = 0;
  entityConnections.forEach((count, entity) => {
    if (count > maxConnections) {
      maxConnections = count;
      mostConnectedEntity = entity;
    }
  });

  return {
    totalRelationships: relationships.length,
    uniqueEntities: entities.size,
    relationshipTypes: relationTypes.size,
    mostConnectedEntity: mostConnectedEntity ? {
      name: mostConnectedEntity,
      connections: maxConnections
    } : null,
    relationshipTypeDistribution: relationTypeDistribution
  };
}