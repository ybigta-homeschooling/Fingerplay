import Home from "./routes/Home";
import Select from "./routes/Select";
import Study from "./routes/Study";
import Cam from "./routes/Cam";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/select" element={<Select />} />
        <Route path="study/:animalId" element={<Study />} />
        <Route path="/cam/:animalId/:level" element={<Cam />} />
      </Routes>
    </Router>
  );
}

export default App;
