import React, {useState} from "react";

export default function QueryBox(){
  const [q, setQ] = useState("");
  const [k, setK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const API_URL = process.env.REACT_APP_API_URL || "/query";

  async function handleSubmit(){
    setLoading(true);
    setResp(null);
    try{
      const r = await fetch(API_URL, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({question: q, k})
      });
      const j = await r.json();
      setResp(j);
    } catch(err) {
      setResp({error: err.message});
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{maxWidth:800}}>
      <textarea value={q} onChange={e=>setQ(e.target.value)} rows={4} style={{width:"100%"}}/>
      <div style={{marginTop:8}}>
        <label>k: </label>
        <input type="number" value={k} onChange={e=>setK(Number(e.target.value))} min={1} max={20}/>
        <button onClick={handleSubmit} disabled={loading || !q} style={{marginLeft:8}}>Ask</button>
      </div>
      {loading && <div>Loading...</div>}
      {resp && resp.answer && (
        <div style={{marginTop:12}}>
          <h3>Answer</h3>
          <div style={{whiteSpace:"pre-wrap"}}>{resp.answer}</div>
          <h4>Sources</h4>
          {resp.sources && resp.sources.map((s,i)=>(
            <div key={i} style={{padding:8, border:"1px solid #eee", marginTop:6}}>
              <div><strong>Metadata:</strong> {JSON.stringify(s.metadata)}</div>
              <div style={{whiteSpace:"pre-wrap"}}>{s.page_content.slice(0,800)}{s.page_content.length>800?"...":""}</div>
            </div>
          ))}
        </div>
      )}
      {resp && resp.error && <div style={{color:"red"}}>{resp.error}</div>}
    </div>
  );
}
