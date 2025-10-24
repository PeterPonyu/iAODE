'use client';

import { useState } from 'react';
import {
  Button,
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
  Input,
  Select,
  Slider,
  Badge,
  Spinner,
  Alert,
} from '@/components/ui';
import { useTheme } from '@/hooks/useTheme';

export default function ComponentsDemoPage() {
  // Test useTheme
  const { theme, effectiveTheme, setTheme, toggleTheme, mounted } = useTheme();
  const [inputValue, setInputValue] = useState('');
  const [selectValue, setSelectValue] = useState('');
  const [sliderValue, setSliderValue] = useState(50);
  const [isLoading, setIsLoading] = useState(false);

  const handleLoadingDemo = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 2000);
  };

  return (
    <div className="container mx-auto p-8 space-y-8">
      <div>
        <h1 className="text-4xl font-bold mb-2">UI Components Demo</h1>
        <p className="text-muted-foreground">
          Showcase of all base UI components with Tailwind v4 styling
        </p>
      </div>
          <div className="card">
            <h2 className="text-xl font-semibold mb-4">🎨 useTheme</h2>
            <div className="space-y-3">
            {mounted ? (
                <div className="space-y-2">
                <p>Current theme: <strong>{theme}</strong></p>
                <p>Effective theme: <strong>{effectiveTheme}</strong></p>
                <div className="p-3 rounded border bg-muted">
                    <p className="text-sm text-muted-foreground">
                    This box uses CSS variables that automatically switch themes!
                    </p>
                </div>
                </div>
            ) : (
                <p className="text-muted-foreground">Loading theme...</p>
            )}
            
            <div className="flex gap-2 flex-wrap">
                <button 
                onClick={toggleTheme} 
                className="btn btn-primary"
                disabled={!mounted}
                >
                {mounted && (effectiveTheme === 'dark' ? '☀️' : '🌙')} Toggle Theme
                </button>
                <button 
                onClick={() => setTheme('light')} 
                className="btn btn-secondary"
                disabled={!mounted}
                >
                ☀️ Light
                </button>
                <button 
                onClick={() => setTheme('dark')} 
                className="btn btn-secondary"
                disabled={!mounted}
                >
                🌙 Dark
                </button>
                <button 
                onClick={() => setTheme('system')} 
                className="btn btn-secondary"
                disabled={!mounted}
                >
                💻 System
                </button>
            </div>
            </div>
        </div>
      {/* Buttons */}
      <Card>
        <CardHeader>
          <CardTitle>Buttons</CardTitle>
          <CardDescription>Different button variants and sizes</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button variant="primary">Primary</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="danger">Danger</Button>
          </div>
          
          <div className="flex flex-wrap gap-2">
            <Button size="sm">Small</Button>
            <Button size="md">Medium</Button>
            <Button size="lg">Large</Button>
          </div>
          
          <div className="space-y-2">
            <Button isLoading>Loading Button</Button>
            <Button disabled>Disabled Button</Button>
            <Button fullWidth>Full Width Button</Button>
          </div>
        </CardContent>
      </Card>

      {/* Cards */}
      <Card hover>
        <CardHeader>
          <CardTitle>Card Component</CardTitle>
          <CardDescription>This is a card with hover effect</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm">
            Cards are containers for content. This one has a hover effect!
          </p>
        </CardContent>
        <CardFooter>
          <Button variant="primary" size="sm">Action</Button>
          <Button variant="ghost" size="sm">Cancel</Button>
        </CardFooter>
      </Card>

      {/* Inputs */}
      <Card>
        <CardHeader>
          <CardTitle>Input Fields</CardTitle>
          <CardDescription>Text inputs with labels and validation</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Input
            label="Name"
            placeholder="Enter your name"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            helperText="This is a helper text"
          />
          
          <Input
            label="Email"
            type="email"
            placeholder="email@example.com"
            error="Invalid email address"
          />
          
          <Input
            placeholder="Disabled input"
            disabled
          />
        </CardContent>
      </Card>

      {/* Select */}
      <Card>
        <CardHeader>
          <CardTitle>Select Dropdown</CardTitle>
          <CardDescription>Dropdown select component</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Select
            label="Choose an option"
            placeholder="Select..."
            value={selectValue}
            onChange={(e) => setSelectValue(e.target.value)}
            options={[
              { value: 'option1', label: 'Option 1' },
              { value: 'option2', label: 'Option 2' },
              { value: 'option3', label: 'Option 3' },
              { value: 'option4', label: 'Disabled Option', disabled: true },
            ]}
            helperText="Select one option from the list"
          />
        </CardContent>
      </Card>

      {/* Slider */}
      <Card>
        <CardHeader>
          <CardTitle>Slider</CardTitle>
          <CardDescription>Range input slider</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Slider
            label="Volume"
            min={0}
            max={100}
            step={1}
            value={sliderValue}
            onChange={(e) => setSliderValue(Number(e.target.value))}
          />
          
          <Slider
            label="Continuity"
            min={0.8}
            max={1.0}
            step={0.01}
            defaultValue={0.95}
            formatValue={(v) => v.toFixed(2)}
          />
        </CardContent>
      </Card>

      {/* Badges */}
      <Card>
        <CardHeader>
          <CardTitle>Badges</CardTitle>
          <CardDescription>Labels and status indicators</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Badge variant="default">Default</Badge>
            <Badge variant="primary">Primary</Badge>
            <Badge variant="success">Success</Badge>
            <Badge variant="warning">Warning</Badge>
            <Badge variant="danger">Danger</Badge>
          </div>
          <div className="flex flex-wrap gap-2 mt-3">
            <Badge size="sm">Small</Badge>
            <Badge size="md">Medium</Badge>
          </div>
        </CardContent>
      </Card>

      {/* Spinner */}
      <Card>
        <CardHeader>
          <CardTitle>Loading Spinner</CardTitle>
          <CardDescription>Loading states and spinners</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-8">
            <Spinner size="sm" label="" />
            <Spinner size="md" />
            <Spinner size="lg" label="Loading data..." />
          </div>
          <div className="mt-6">
            <Button onClick={handleLoadingDemo} isLoading={isLoading}>
              {isLoading ? 'Loading...' : 'Trigger Loading'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Alerts */}
      <Card>
        <CardHeader>
          <CardTitle>Alerts</CardTitle>
          <CardDescription>Messages and notifications</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert variant="info" title="Information">
            This is an informational message.
          </Alert>
          
          <Alert variant="success" title="Success!">
            Your changes have been saved successfully.
          </Alert>
          
          <Alert variant="warning" title="Warning">
            Please review your input before proceeding.
          </Alert>
          
          <Alert variant="error" title="Error">
            An error occurred while processing your request.
          </Alert>
          
          <Alert variant="info" title="Dismissible" onClose={() => alert('Closed!')}>
            This alert can be closed.
          </Alert>
        </CardContent>
      </Card>
    </div>
  );
}