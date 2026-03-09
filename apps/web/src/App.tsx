import { Route, Routes } from "react-router-dom";

import AppShell from "./components/AppShell";
import JobsProvider from "./components/JobsProvider";
import PersistentPageOutlet from "./components/PersistentPageOutlet";

export default function App() {
  return (
    <Routes>
      <Route
        element={
          <JobsProvider>
            <AppShell />
          </JobsProvider>
        }
      >
        <Route path="*" element={<PersistentPageOutlet />} />
      </Route>
    </Routes>
  );
}
