import { useCallback, useEffect, useState, useMemo } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  ConnectionLineType,
  Panel,
} from 'reactflow';
import 'reactflow/dist/style.css';
import './RelationshipGraph.css';
import { buildGraphData, optimizeLayout, getGraphStats } from '../utils/graphBuilder';

const RelationshipGraph = ({ relationships, entities = [] }) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [showStats, setShowStats] = useState(false);

  // Calculate graph data and statistics
  const { graphData, stats } = useMemo(() => {
    const data = buildGraphData(relationships, entities);
    const statistics = getGraphStats(relationships);
    
    // Apply layout optimization
    if (data.nodes.length > 0) {
      const optimizedNodes = optimizeLayout(data.nodes, data.edges);
      data.nodes = optimizedNodes;
    }
    
    // Ensure stats has all required properties with defaults
    const safeStats = {
      totalRelationships: 0,
      uniqueEntities: 0,
      relationshipTypes: 0,
      mostConnectedEntity: null,
      relationshipTypeDistribution: {},
      ...statistics
    };
    
    return {
      graphData: data,
      stats: safeStats
    };
  }, [relationships, entities]);

  // Update nodes and edges when data changes
  useEffect(() => {
    setNodes(graphData.nodes);
    setEdges(graphData.edges);
  }, [graphData, setNodes, setEdges]);

  // Handle node clicks
  const onNodeClick = useCallback((event, node) => {
    setSelectedNode(node);
    setSelectedEdge(null);
    
    // Highlight connected edges
    setEdges(edges => 
      edges.map(edge => ({
        ...edge,
        animated: edge.source === node.id || edge.target === node.id,
        style: {
          ...edge.style,
          strokeWidth: edge.source === node.id || edge.target === node.id ? 3 : 2,
          opacity: edge.source === node.id || edge.target === node.id ? 1 : 0.6,
        }
      }))
    );
  }, [setEdges]);

  // Handle edge clicks
  const onEdgeClick = useCallback((event, edge) => {
    setSelectedEdge(edge);
    setSelectedNode(null);
    
    // Reset edge highlighting
    setEdges(edges => 
      edges.map(e => ({
        ...e,
        animated: e.id === edge.id,
        style: {
          ...e.style,
          strokeWidth: e.id === edge.id ? 3 : 2,
          opacity: 1,
        }
      }))
    );
  }, [setEdges]);

  // Clear selection when clicking on background
  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setSelectedEdge(null);
    
    // Reset all highlighting
    setEdges(edges => 
      edges.map(edge => ({
        ...edge,
        animated: false,
        style: {
          ...edge.style,
          strokeWidth: 2,
          opacity: 1,
        }
      }))
    );
  }, [setEdges]);

  // Handle connection creation (if needed for future features)
  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Fit graph to view
  const fitView = useCallback(() => {
    // This would be handled by ReactFlow's fitView function
    // We'll implement this with the ReactFlow instance ref if needed
  }, []);

  if (relationships.length === 0) {
    return (
      <div className="relationship-graph-empty">
        <div className="empty-state">
          <h3>No Relationships Found</h3>
          <p>Submit some biomedical text to see the knowledge graph visualization.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relationship-graph-container">
      <div className="graph-header">
        <h3>Knowledge Graph Visualization</h3>
        <div className="graph-controls">
          <button 
            className="stats-toggle"
            onClick={() => setShowStats(!showStats)}
          >
            {showStats ? 'Hide' : 'Show'} Stats
          </button>
        </div>
      </div>

      <div className="reactflow-wrapper">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onEdgeClick={onEdgeClick}
          onPaneClick={onPaneClick}
          connectionLineType={ConnectionLineType.SmoothStep}
          fitView
          fitViewOptions={{ padding: 0.2 }}
        >
          <Controls />
          <MiniMap 
            nodeStrokeColor="#333"
            nodeColor="#3b82f6"
            nodeBorderRadius={2}
            maskColor="rgba(255, 255, 255, 0.8)"
            style={{ backgroundColor: '#f8fafc' }}
          />
          <Background variant="dots" gap={12} size={1} />
          
          {/* Info Panel */}
          <Panel position="top-left" className="graph-info-panel">
            {selectedNode && (
              <div className="selection-info">
                <h4>Entity: {selectedNode.data.label}</h4>
                <div className="entity-details">
                  <p><strong>Connections:</strong> {selectedNode.data.connectionCount}</p>
                  {selectedNode.data.cui && (
                    <p><strong>CUI:</strong> {selectedNode.data.cui}</p>
                  )}
                  {selectedNode.data.canonicalName && (
                    <p><strong>Name:</strong> {selectedNode.data.canonicalName}</p>
                  )}
                  {selectedNode.data.semanticTypes && (
                    <p><strong>Types:</strong> {selectedNode.data.semanticTypes.join(', ')}</p>
                  )}
                </div>
              </div>
            )}
            
            {selectedEdge && (
              <div className="selection-info">
                <h4>Relationship</h4>
                <div className="relationship-details">
                  <p><strong>Type:</strong> {selectedEdge.label}</p>
                  <p><strong>From:</strong> {nodes.find(n => n.id === selectedEdge.source)?.data.label}</p>
                  <p><strong>To:</strong> {nodes.find(n => n.id === selectedEdge.target)?.data.label}</p>
                </div>
              </div>
            )}
          </Panel>

          {/* Statistics Panel */}
          {showStats && (
            <Panel position="top-right" className="graph-stats-panel">
              <div className="stats-content">
                <h4>Graph Statistics</h4>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-label">Entities:</span>
                    <span className="stat-value">{stats.uniqueEntities}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Relationships:</span>
                    <span className="stat-value">{stats.totalRelationships}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Relation Types:</span>
                    <span className="stat-value">{stats.relationshipTypes}</span>
                  </div>
                  {stats.mostConnectedEntity && (
                    <div className="stat-item">
                      <span className="stat-label">Most Connected:</span>
                      <span className="stat-value">{stats.mostConnectedEntity.name}</span>
                    </div>
                  )}
                </div>
                
                {/* Relationship Type Distribution */}
                <div className="relation-types">
                  <h5>Relationship Types:</h5>
                  {Object.entries(stats.relationshipTypeDistribution || {}).map(([type, count]) => (
                    <div key={type} className="relation-type-item">
                      <span className="relation-type">{type}</span>
                      <span className="relation-count">({count})</span>
                    </div>
                  ))}
                </div>
              </div>
            </Panel>
          )}
        </ReactFlow>
      </div>

      {/* Legend */}
      <div className="graph-legend">
        <h5>Legend</h5>
        <div className="legend-items">
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#ef4444' }}></div>
            <span>High connections (5+)</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#f97316' }}></div>
            <span>Medium connections (3-4)</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#eab308' }}></div>
            <span>Some connections (2)</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#3b82f6' }}></div>
            <span>Few connections (1)</span>
          </div>
        </div>
        
        <div className="legend-instructions">
          <p><strong>Instructions:</strong></p>
          <ul>
            <li>Click nodes to highlight connections</li>
            <li>Click edges to see relationship details</li>
            <li>Drag nodes to reposition them</li>
            <li>Use controls to pan, zoom, and fit view</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default RelationshipGraph;