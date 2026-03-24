# Buildify 🏗️

Buildify is an AI-powered architectural floor plan generator that produces true-to-scale, IRC-compliant (International Residential Code) home layouts using a combination of a Mixture of Experts (MOE) model, HouseGAN++ spatial layout generation, and deterministic zone-based packing.

## 🚀 Features

- **AI-Driven Floor Plans**: Instant generation of multiple layout variations based on user constraints (beds, baths, sqft, style).
- **Zero-Gap US Architectural Layouts**: Plans are generated using realistic American residential depth/width ratios (e.g., 68W × 60D) with no gaps between rooms.
- **Charrette Board UI**: Compare generated variants side-by-side with color-coded 2D minimaps, key statistics, and footprint dimensions.
- **Interactive 3D Walkthroughs**: Visualize exteriors with clean, white architectural models.
- **Advanced Constraint Engine**: Granular control over primary suites, home offices, garage sizes, outdoor patios, etc.

## 🧠 Architecture Setup

Buildify uses a multi-stage hybrid generation pipeline to produce sensible floor plans:

1. **Stage 1 (Build)**: A deterministic module translates user constraints into a precise list of required rooms, using standard IRC sizes.
2. **Stage 2 (Size)**: The locally trained MOE model outputs expert weights that intelligently scale room dimensions based on the target square footage and architectural style.
3. **Stage 3 (Place)**: Rooms are placed into a spatial layout. Buildify integrates **HouseGAN++** for graph-based topological placement. If unavailable, it falls back to a robust, zone-based US architectural solver (Entry, Social, Private, Outdoor bands).
4. **Stage 4 (Validate)**: An IRC compliance checker ensures all rooms meet minimum habitability dimensions (e.g., 70 sq ft minimum for bedrooms).
5. **Stage 5 (Refine)**: A grid-snapping and tight-packing algorithm eliminates gaps and resolves any overlaps, forcing dimensions onto a 2ft construction grid.

## 🛠️ Tech Stack

**Frontend:**
- React (Vite)
- TypeScript
- Three.js / React Three Fiber (for 3D architectural visualization)
- Vanilla CSS (custom design system)

**Backend:**
- FastAPI (Python)
- PyTorch (for MOE inference)
- HouseGAN++ 

## 🏃‍♂️ Running Locally

1. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   python3 -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 📚 About the Project

Buildify was created to bring instant, conceptual architectural design into the hands of users. By combining deep learning with strict, rule-based geometric packing, it resolves a common issue with generative AI in architecture: hallucinated, unbuildable shapes. Every output from Buildify is a structurally sound, grid-aligned arrangement.

## 📝 License
This project is open-source.
