import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { MainLayout } from "@/components/layout/MainLayout";
import Dashboard from "@/pages/Dashboard";
import Hospitals from "@/pages/Hospitals";
import HospitalDetail from "@/pages/HospitalDetail";
import FederatedLearning from "@/pages/FederatedLearning";
import FLWorkflow from "@/pages/FLWorkflow";
import IoMTMonitor from "@/pages/IoMTMonitor";
import Explainability from "@/pages/Explainability";
import ModelInsights from "@/pages/ModelInsights";
import ProductStory from "@/pages/ProductStory";
import Settings from "@/pages/Settings";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <MainLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/hospitals" element={<Hospitals />} />
            <Route path="/hospitals/:id" element={<HospitalDetail />} />
            <Route path="/federated-learning" element={<FederatedLearning />} />
            <Route path="/fl-workflow" element={<FLWorkflow />} />
            <Route path="/iomt-monitor" element={<IoMTMonitor />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/model-insights" element={<ModelInsights />} />
            <Route path="/product-story" element={<ProductStory />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </MainLayout>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
