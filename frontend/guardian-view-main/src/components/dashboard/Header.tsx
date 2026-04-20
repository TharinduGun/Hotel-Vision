import { Bell, Settings, Shield, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Header = () => {
  const currentDate = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="flex h-16 items-center justify-between px-4 md:px-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" className="md:hidden">
            <Menu className="w-5 h-5" />
          </Button>
          
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary/10 border border-primary/20">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-lg font-bold tracking-tight">
                <span className="text-gradient">SecureView</span>
              </h1>
              <p className="text-[10px] text-muted-foreground font-mono uppercase tracking-wider">
                Hotel CCTV Dashboard
              </p>
            </div>
          </div>
        </div>

        <div className="hidden md:block">
          <p className="text-sm text-muted-foreground font-mono">{currentDate}</p>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="icon" className="relative">
            <Bell className="w-5 h-5" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-destructive animate-pulse" />
          </Button>
          <Button variant="ghost" size="icon">
            <Settings className="w-5 h-5" />
          </Button>
          <div className="ml-2 hidden sm:flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-muted overflow-hidden border-2 border-primary/30">
              <img
                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=100&h=100&fit=crop&crop=face"
                alt="User"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
