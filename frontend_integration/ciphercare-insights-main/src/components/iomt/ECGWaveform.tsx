import React, { useEffect, useRef, useState } from 'react';
import { Maximize2, Minimize2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ECGPoint } from '@/lib/iomt-simulator';
import { cn } from '@/lib/utils';

interface ECGWaveformProps {
  data: ECGPoint[];
  height?: number;
  className?: string;
  showControls?: boolean;
  patientId?: string;
}

export const ECGWaveform: React.FC<ECGWaveformProps> = ({
  data,
  height = 200,
  className,
  showControls = true,
  patientId = 'A-2341',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawWaveform = () => {
      const width = canvas.width;
      const h = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, h);
      
      // Draw grid
      ctx.strokeStyle = 'hsl(var(--border))';
      ctx.lineWidth = 0.5;
      ctx.setLineDash([2, 4]);
      
      // Horizontal grid lines
      const centerY = h / 2;
      for (let i = 0; i < 5; i++) {
        const y = (h / 5) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      
      // Vertical grid lines
      for (let i = 0; i < 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
      
      // Draw ECG line
      if (data.length > 1) {
        ctx.strokeStyle = 'hsl(var(--success))';
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        const minValue = -0.5;
        const maxValue = 1.2;
        const range = maxValue - minValue;
        const scaleY = h / range;
        const scaleX = width / data.length;
        
        ctx.beginPath();
        data.forEach((point, index) => {
          const x = index * scaleX;
          const y = h - ((point.value - minValue) * scaleY);
          
          if (index === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
        
        // Add glow effect
        ctx.shadowBlur = 8;
        ctx.shadowColor = 'hsl(var(--success))';
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    };

    drawWaveform();
    
    // Smooth animation loop
    const animate = () => {
      drawWaveform();
      animationFrameRef.current = requestAnimationFrame(animate);
    };
    
    animationFrameRef.current = requestAnimationFrame(animate);
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [data, height]);

  const handleFullscreen = () => {
    if (!isFullscreen) {
      const canvas = canvasRef.current;
      if (canvas?.requestFullscreen) {
        canvas.requestFullscreen();
        setIsFullscreen(true);
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
        setIsFullscreen(false);
      }
    }
  };

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  return (
    <div className={cn("relative", className)}>
      <div className="card-medical p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
            <span className="text-sm font-medium text-foreground">ECG Waveform - Patient {patientId}</span>
          </div>
          {showControls && (
            <Button
              variant="ghost"
              size="icon"
              onClick={handleFullscreen}
              className="h-8 w-8"
            >
              {isFullscreen ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </Button>
          )}
        </div>
        <div className="bg-foreground/5 rounded-lg p-2">
          <canvas
            ref={canvasRef}
            width={800}
            height={height}
            className="w-full h-full"
            style={{ maxHeight: `${height}px` }}
          />
        </div>
        <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
          <span>25mm/s • 10mm/mV</span>
          <span>Lead II</span>
        </div>
      </div>
    </div>
  );
};





