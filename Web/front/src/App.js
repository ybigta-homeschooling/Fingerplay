import Home from "./routes/Home";
import Select from "./routes/Select";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/select" element={<Select />} />
        <Route path="/" element={<Home />} />
      </Routes>
    </Router>
  );
}

export default App;
