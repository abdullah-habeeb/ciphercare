/**
 * UseCaseForm Component
 * Dynamic form generator for hospital-specific use cases
 */

import React, { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface FormField {
  name: string;
  label: string;
  type: 'number' | 'text' | 'select' | 'file';
  placeholder?: string;
  options?: string[];
  min?: number;
  max?: number;
  required?: boolean;
}

export interface UseCaseFormProps {
  schema: FormField[];
  onSubmit: (data: Record<string, any>) => void;
  isLoading?: boolean;
}

export const UseCaseForm: React.FC<UseCaseFormProps> = ({ schema, onSubmit, isLoading = false }) => {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validateField = (field: FormField, value: any): string | null => {
    if (field.required && (!value || value === '')) {
      return `${field.label} is required`;
    }
    
    if (field.type === 'number' && value !== '' && value !== undefined) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) {
        return `${field.label} must be a number`;
      }
      if (field.min !== undefined && numValue < field.min) {
        return `${field.label} must be at least ${field.min}`;
      }
      if (field.max !== undefined && numValue > field.max) {
        return `${field.label} must be at most ${field.max}`;
      }
    }
    
    return null;
  };

  const handleChange = (name: string, value: any) => {
    setFormData(prev => ({ ...prev, [name]: value }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[name];
        return newErrors;
      });
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate all fields
    const newErrors: Record<string, string> = {};
    schema.forEach(field => {
      const error = validateField(field, formData[field.name]);
      if (error) {
        newErrors[field.name] = error;
      }
    });
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    // Submit if valid
    onSubmit(formData);
  };

  const renderField = (field: FormField) => {
    const hasError = !!errors[field.name];
    const value = formData[field.name] ?? '';

    switch (field.type) {
      case 'number':
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name} className={cn(hasError && "text-destructive")}>
              {field.label} {field.required && <span className="text-destructive">*</span>}
            </Label>
            <Input
              id={field.name}
              type="number"
              placeholder={field.placeholder}
              value={value}
              onChange={(e) => handleChange(field.name, e.target.value)}
              min={field.min}
              max={field.max}
              className={cn(hasError && "border-destructive")}
              disabled={isLoading}
            />
            {hasError && (
              <p className="text-sm text-destructive">{errors[field.name]}</p>
            )}
          </div>
        );

      case 'select':
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name} className={cn(hasError && "text-destructive")}>
              {field.label} {field.required && <span className="text-destructive">*</span>}
            </Label>
            <Select
              value={value}
              onValueChange={(val) => handleChange(field.name, val)}
              disabled={isLoading}
            >
              <SelectTrigger className={cn(hasError && "border-destructive")}>
                <SelectValue placeholder={field.placeholder || `Select ${field.label}`} />
              </SelectTrigger>
              <SelectContent>
                {field.options?.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {hasError && (
              <p className="text-sm text-destructive">{errors[field.name]}</p>
            )}
          </div>
        );

      case 'file':
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name} className={cn(hasError && "text-destructive")}>
              {field.label} {field.required && <span className="text-destructive">*</span>}
            </Label>
            <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
              <p className="text-sm text-muted-foreground mb-2">
                {field.placeholder || 'Image upload (placeholder)'}
              </p>
              <Button type="button" variant="outline" disabled>
                Select File (Disabled)
              </Button>
            </div>
            {hasError && (
              <p className="text-sm text-destructive">{errors[field.name]}</p>
            )}
          </div>
        );

      default:
        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name} className={cn(hasError && "text-destructive")}>
              {field.label} {field.required && <span className="text-destructive">*</span>}
            </Label>
            <Input
              id={field.name}
              type="text"
              placeholder={field.placeholder}
              value={value}
              onChange={(e) => handleChange(field.name, e.target.value)}
              className={cn(hasError && "border-destructive")}
              disabled={isLoading}
            />
            {hasError && (
              <p className="text-sm text-destructive">{errors[field.name]}</p>
            )}
          </div>
        );
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {schema.map(field => renderField(field))}
      <Button
        type="submit"
        className="w-full"
        disabled={isLoading}
      >
        {isLoading ? (
          <>
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Processing...
          </>
        ) : (
          'Run Prediction'
        )}
      </Button>
    </form>
  );
};

