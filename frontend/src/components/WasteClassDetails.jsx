import { Package, GlassWater, Cog, FileText, Milk, Trash2, Factory, Recycle, Leaf, Sparkles } from "lucide-react";

const wasteClasses = {
  cardboard: {
    name: "Cardboard",
    icon: Package,
    color: "from-amber-500/20 to-amber-600/10",
    borderColor: "border-amber-500/30",
    iconBg: "bg-amber-500/10",
    description: "Corrugated boxes, packaging materials, and cardboard sheets.",
    industries: ["Packaging Industry", "Paper Mills", "Furniture Manufacturing"],
    uses: ["New cardboard boxes", "Paperboard products", "Egg cartons", "Building insulation"]
  },
  glass: {
    name: "Glass",
    icon: GlassWater,
    color: "from-sky-500/20 to-sky-600/10",
    borderColor: "border-sky-500/30",
    iconBg: "bg-sky-500/10",
    description: "Bottles, jars, and glass containers of all colors.",
    industries: ["Beverage Industry", "Construction", "Art & Crafts"],
    uses: ["New glass containers", "Fiberglass insulation", "Road construction", "Decorative tiles"]
  },
  metal: {
    name: "Metal",
    icon: Cog,
    color: "from-slate-400/20 to-slate-500/10",
    borderColor: "border-slate-400/30",
    iconBg: "bg-slate-400/10",
    description: "Aluminum cans, steel containers, and metal scraps.",
    industries: ["Automotive Industry", "Construction", "Electronics"],
    uses: ["Car parts", "Building materials", "New cans", "Machinery components"]
  },
  paper: {
    name: "Paper",
    icon: FileText,
    color: "from-emerald-500/20 to-emerald-600/10",
    borderColor: "border-emerald-500/30",
    iconBg: "bg-emerald-500/10",
    description: "Newspapers, magazines, office paper, and paper bags.",
    industries: ["Publishing", "Packaging", "Hygiene Products"],
    uses: ["Recycled paper", "Tissue products", "Newsprint", "Cardboard boxes"]
  },
  plastic: {
    name: "Plastic",
    icon: Milk,
    color: "from-blue-500/20 to-blue-600/10",
    borderColor: "border-blue-500/30",
    iconBg: "bg-blue-500/10",
    description: "Bottles, containers, bags, and plastic packaging.",
    industries: ["Textile Industry", "Furniture", "Construction"],
    uses: ["Polyester fabric", "Plastic lumber", "Containers", "Playground equipment"]
  },
  trash: {
    name: "Trash",
    icon: Trash2,
    color: "from-rose-500/20 to-rose-600/10",
    borderColor: "border-rose-500/30",
    iconBg: "bg-rose-500/10",
    description: "Non-recyclable waste that requires proper disposal.",
    industries: ["Waste Management", "Energy Production"],
    uses: ["Waste-to-energy plants", "Landfill management", "Composting (organic)", "Incineration"]
  }
};

const WasteClassDetails = ({ predictedClass }) => {
  if (!predictedClass) {
    return (
      <div className="mt-8 p-6 bg-card border border-border rounded-xl text-center">
        <div className="p-3 bg-muted rounded-full w-fit mx-auto mb-4">
          <Sparkles className="w-6 h-6 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-medium text-foreground mb-2">Recycling Information</h3>
        <p className="text-sm text-muted-foreground">
          Upload an image to see detailed recycling information and industry uses for the identified waste type.
        </p>
      </div>
    );
  }

  const waste = wasteClasses[predictedClass.toLowerCase()];

  if (!waste) {
    return null;
  }

  const IconComponent = waste.icon;

  return (
    <div className="mt-8">
      <div className={`bg-gradient-to-br ${waste.color} border ${waste.borderColor} rounded-xl p-6 transition-all duration-500 animate-fade-in`}>
        <div className="flex items-center gap-3 mb-4">
          <div className={`p-3 ${waste.iconBg} rounded-xl`}>
            <IconComponent className="w-6 h-6 text-foreground" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-foreground">{waste.name}</h3>
            <p className="text-sm text-muted-foreground">Recycling & Industry Information</p>
          </div>
        </div>
        
        <p className="text-muted-foreground mb-6">{waste.description}</p>
        
        <div className="grid gap-6 md:grid-cols-2">
          {/* Industries Section */}
          <div className="bg-card/50 rounded-lg p-4 border border-border/50">
            <div className="flex items-center gap-2 mb-3">
              <Factory className="w-4 h-4 text-primary" />
              <span className="text-sm font-semibold text-foreground">Where It's Used</span>
            </div>
            <div className="space-y-2">
              {waste.industries.map((industry) => (
                <div
                  key={industry}
                  className="flex items-center gap-2 px-3 py-2 bg-secondary rounded-lg"
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                  <span className="text-sm text-secondary-foreground">{industry}</span>
                </div>
              ))}
            </div>
          </div>
          
          {/* Recycled Into Section */}
          <div className="bg-card/50 rounded-lg p-4 border border-border/50">
            <div className="flex items-center gap-2 mb-3">
              <Recycle className="w-4 h-4 text-primary" />
              <span className="text-sm font-semibold text-foreground">Recycled Into</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {waste.uses.map((use) => (
                <span
                  key={use}
                  className="px-3 py-1.5 bg-primary/10 text-primary text-sm rounded-full border border-primary/20"
                >
                  {use}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Eco tip */}
        <div className="mt-6 flex items-start gap-3 p-4 bg-primary/5 rounded-lg border border-primary/10">
          <Leaf className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
          <p className="text-sm text-muted-foreground">
            {predictedClass.toLowerCase() === 'trash' 
              ? "While this item may not be recyclable, consider if any parts can be separated for recycling or if it can be repurposed."
              : `Recycling ${waste.name.toLowerCase()} helps conserve natural resources and reduces landfill waste. Make sure to clean and sort before recycling!`
            }
          </p>
        </div>
      </div>
    </div>
  );
};

export default WasteClassDetails;
