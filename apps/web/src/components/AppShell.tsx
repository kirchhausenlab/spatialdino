import { Outlet } from "react-router-dom";

import Sidebar from "./Sidebar";
import TopNav from "./TopNav";

export default function AppShell() {
  return (
    <div className="appShell">
      <TopNav />
      <div className="appBody">
        <Sidebar />
        <main className="appMain">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
