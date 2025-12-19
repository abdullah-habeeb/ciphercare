import React from 'react';
import { Link } from 'react-router-dom';
import { HOSPITALS } from '@/lib/constants';
import { HospitalCard } from '@/components/ui/hospital-card';
import { 
  Building2, 
  Search,
  Filter,
  Grid,
  List
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

const Hospitals: React.FC = () => {
  const [view, setView] = React.useState<'grid' | 'list'>('grid');
  const [search, setSearch] = React.useState('');

  const filteredHospitals = HOSPITALS.filter(h => 
    h.name.toLowerCase().includes(search.toLowerCase()) ||
    h.specialty.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-display font-bold text-foreground flex items-center gap-3">
            <Building2 className="w-8 h-8 text-primary" />
            Hospital Network
          </h1>
          <p className="text-muted-foreground mt-1">
            Manage and monitor all connected healthcare institutions
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search hospitals..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="icon">
            <Filter className="w-4 h-4" />
          </Button>
          <div className="flex border border-border rounded-lg overflow-hidden">
            <Button 
              variant={view === 'grid' ? 'secondary' : 'ghost'} 
              size="icon"
              onClick={() => setView('grid')}
            >
              <Grid className="w-4 h-4" />
            </Button>
            <Button 
              variant={view === 'list' ? 'secondary' : 'ghost'} 
              size="icon"
              onClick={() => setView('list')}
            >
              <List className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Stats Banner */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Total Hospitals</p>
          <p className="text-2xl font-bold font-mono text-foreground">5</p>
        </div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Active Connections</p>
          <p className="text-2xl font-bold font-mono text-success">5</p>
        </div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Combined Samples</p>
          <p className="text-2xl font-bold font-mono text-foreground">193K</p>
        </div>
        <div className="card-medical p-4">
          <p className="text-sm text-muted-foreground">Avg. AUROC</p>
          <p className="text-2xl font-bold font-mono text-primary">0.909</p>
        </div>
      </div>

      {/* Hospital Grid */}
      <div className={view === 'grid' 
        ? "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4" 
        : "space-y-4"
      }>
        {filteredHospitals.map((hospital) => (
          <HospitalCard key={hospital.id} hospital={hospital} />
        ))}
      </div>

      {filteredHospitals.length === 0 && (
        <div className="text-center py-12">
          <Building2 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground">No hospitals found</h3>
          <p className="text-muted-foreground">Try adjusting your search criteria</p>
        </div>
      )}
    </div>
  );
};

export default Hospitals;
