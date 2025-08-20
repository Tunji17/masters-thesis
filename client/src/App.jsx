import { useState } from 'react';
import { medicalApi } from './api/medicalApi';
import RelationshipGraph from './components/RelationshipGraph';
import './App.css';

function App() {
  const [clinicalText, setClinicalText] = useState(`Patient presents with chest pain and shortness of breath. He reports experiencing sharp, stabbing pain in the left side of his chest that worsens with deep inspiration. The pain started approximately 2 hours ago and is associated with difficulty breathing.

Patient has a history of hypertension and diabetes mellitus type 2. He takes metformin for diabetes and lisinopril for hypertension. Physical examination reveals tachycardia with heart rate of 110 bpm and blood pressure of 150/90 mmHg. 

Chest X-ray shows no evidence of pneumothorax or pleural effusion. ECG demonstrates ST-elevation in leads II, III, and aVF, consistent with inferior wall myocardial infarction. Cardiac enzymes are elevated with troponin I at 2.5 ng/mL.

Diagnosis: Acute ST-elevation myocardial infarction (STEMI) involving the inferior wall. Patient was started on dual antiplatelet therapy with aspirin and clopidogrel, along with atorvastatin for cholesterol management.`);
  const [extractionResults, setExtractionResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!clinicalText.trim()) {
      setError('Please enter some clinical text to analyze.');
      return;
    }

    setLoading(true);
    setError(null);
    setExtractionResults(null);

    try {
      // Step 1: Extract entities and relationships, store in Neo4j
      console.log('Extracting entities and relationships...');
      const extractionResult = await medicalApi.extractFullPipeline(clinicalText, {
        threshold: 0.5,
        maxTokens: 512,
        storeInGraph: true,
      });
      
      setExtractionResults(extractionResult);
      console.log('Extraction completed:', extractionResult);
    } catch (err) {
      console.error('Processing failed:', err);
      setError(
        err.response?.data?.detail || 
        err.message || 
        'An error occurred while processing the clinical text.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setClinicalText('');
    setExtractionResults(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Medical Graph Extraction</h1>
        <p>Enter Biomedical text to extract entities and relationships</p>
      </header>

      <main className="app-main">
        <form onSubmit={handleSubmit} className="input-section">
          <div className="textarea-container">
            <label htmlFor="clinical-text">Biomedical Text:</label>
            <textarea
              id="clinical-text"
              value={clinicalText}
              onChange={(e) => setClinicalText(e.target.value)}
              placeholder="Enter Biomedical notes here... (e.g., 'Patient presents with chest pain and shortness of breath. Diagnosed with myocardial infarction.')"
              rows={8}
              disabled={loading}
            />
          </div>
          
          <div className="button-group">
            <button type="submit" disabled={loading || !clinicalText.trim()}>
              {loading ? 'Processing...' : 'Analyze Text'}
            </button>
            <button type="button" onClick={handleClear} disabled={loading}>
              Clear
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}

        {extractionResults && (
          <div className="results-section">
            <h2>Extraction Results</h2>
            
            <div className="results-summary">
              <div className="summary-item">
                <span className="label">Note ID:</span>
                <span className="value">{extractionResults.note_id}</span>
              </div>
              <div className="summary-item">
                <span className="label">Stored in Graph:</span>
                <span className="value">{extractionResults.stored ? 'Yes' : 'No'}</span>
              </div>
            </div>


            <RelationshipGraph 
              relationships={extractionResults.relationships}
              entities={extractionResults.entities}
            />
          </div>
        )}

      </main>
    </div>
  );
}

export default App;