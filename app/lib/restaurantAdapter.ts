import type {
  MenuItem,
  Restaurant,
  RestaurantMealType,
} from "@/app/types/restaurant";

export type ExternalRestaurant = {
  id: string;
  name: string;
  category?: string;
  cuisine?: string;
  address?: string;
  mapUrl?: string;
  menu?: ExternalMenuItem[];
};

export type ExternalMenuItem = {
  id: string;
  name: string;
  calories?: number;
  protein?: number;
  carbs?: number;
  fat?: number;
  fiber?: number;
};

export type GooglePlace = {
  id: string;
  displayName?: {
    text?: string;
  };
  formattedAddress?: string;
  primaryType?: string;
};

export function adaptRestaurant(
  externalRestaurant: ExternalRestaurant
): Restaurant {
  const menu = (externalRestaurant.menu ?? []).map(adaptMenuItem);

  const averageCalories =
    menu.length === 0
      ? 650
      : Math.round(
          menu.reduce((sum, item) => sum + item.calories, 0) / menu.length
        );

  return {
    id: externalRestaurant.id,
    name: externalRestaurant.name,
    category: externalRestaurant.category ?? "Restaurant",
    cuisine: externalRestaurant.cuisine ?? "Unknown",
    mealType: inferMealType(menu),
    averageCalories,
    proteinScore: inferProteinScore(menu),
    healthScore: inferHealthScore(menu),
    matchScore: 0,
    tags: inferRestaurantTags(
      externalRestaurant.cuisine ?? "Unknown",
      externalRestaurant.category ?? "Restaurant"
    ),
    menu,
  };
}

export function adaptMenuItem(item: ExternalMenuItem): MenuItem {
  return {
    id: item.id,
    name: item.name,
    calories: item.calories ?? 0,
    protein: item.protein ?? 0,
    carbs: item.carbs ?? 0,
    fat: item.fat ?? 0,
    fiber: item.fiber ?? 0,
    tags: [],
  };
}

export function adaptGooglePlaceToRestaurant(place: GooglePlace): Restaurant {
  const name = place.displayName?.text ?? "Unknown Restaurant";
  const category = formatGooglePlaceType(place.primaryType);
  const cuisine = inferCuisineFromText(`${name} ${category}`);

  return adaptRestaurant({
    id: createSafeRestaurantId(place.id),
    name,
    category,
    cuisine,
    address: place.formattedAddress,
    mapUrl: createGoogleMapsSearchUrl(name, place.formattedAddress ?? ""),
    menu: createEstimatedMenuForCuisine(cuisine),
  });
}

export function adaptGooglePlacesToRestaurants(
  places: GooglePlace[]
): Restaurant[] {
  return places.map(adaptGooglePlaceToRestaurant);
}

function createSafeRestaurantId(value: string): string {
  return value
    .replace("places/", "")
    .replace(/[^a-zA-Z0-9-_]/g, "-")
    .toLowerCase();
}

function createEstimatedMenuForCuisine(cuisine: string): ExternalMenuItem[] {
  const normalizedCuisine = cuisine.toLowerCase();

  if (normalizedCuisine === "japanese" || normalizedCuisine === "sushi") {
    return [
      {
        id: "estimated-salmon-sushi-set",
        name: "Estimated Salmon Sushi Set",
        calories: 620,
        protein: 36,
        carbs: 68,
        fat: 18,
        fiber: 4,
      },
      {
        id: "estimated-tuna-rice-bowl",
        name: "Estimated Tuna Rice Bowl",
        calories: 580,
        protein: 42,
        carbs: 62,
        fat: 12,
        fiber: 5,
      },
    ];
  }

  if (normalizedCuisine === "indian") {
    return [
      {
        id: "estimated-chicken-curry",
        name: "Estimated Chicken Curry",
        calories: 720,
        protein: 38,
        carbs: 76,
        fat: 26,
        fiber: 8,
      },
      {
        id: "estimated-tandoori-chicken",
        name: "Estimated Tandoori Chicken",
        calories: 560,
        protein: 48,
        carbs: 32,
        fat: 18,
        fiber: 5,
      },
    ];
  }

  if (normalizedCuisine === "burger") {
    return [
      {
        id: "estimated-grilled-chicken-burger",
        name: "Estimated Grilled Chicken Burger",
        calories: 650,
        protein: 44,
        carbs: 58,
        fat: 22,
        fiber: 5,
      },
      {
        id: "estimated-classic-burger",
        name: "Estimated Classic Burger",
        calories: 860,
        protein: 38,
        carbs: 72,
        fat: 42,
        fiber: 4,
      },
    ];
  }

  if (normalizedCuisine === "healthy") {
    return [
      {
        id: "estimated-chicken-protein-bowl",
        name: "Estimated Chicken Protein Bowl",
        calories: 520,
        protein: 45,
        carbs: 48,
        fat: 14,
        fiber: 9,
      },
      {
        id: "estimated-tofu-fiber-bowl",
        name: "Estimated Tofu Fiber Bowl",
        calories: 480,
        protein: 28,
        carbs: 55,
        fat: 12,
        fiber: 14,
      },
    ];
  }

  return [
    {
      id: "estimated-balanced-meal",
      name: "Estimated Balanced Meal",
      calories: 650,
      protein: 35,
      carbs: 65,
      fat: 22,
      fiber: 6,
    },
    {
      id: "estimated-light-meal",
      name: "Estimated Light Meal",
      calories: 520,
      protein: 30,
      carbs: 55,
      fat: 16,
      fiber: 7,
    },
  ];
}

function inferCuisineFromText(text: string): string {
  const lowerText = text.toLowerCase();

  if (
    lowerText.includes("sushi") ||
    lowerText.includes("japanese") ||
    lowerText.includes("ramen")
  ) {
    return "Japanese";
  }

  if (
    lowerText.includes("indian") ||
    lowerText.includes("curry") ||
    lowerText.includes("tandoori") ||
    lowerText.includes("biryani")
  ) {
    return "Indian";
  }

  if (
    lowerText.includes("burger") ||
    lowerText.includes("grill") ||
    lowerText.includes("five guys")
  ) {
    return "Burger";
  }

  if (
    lowerText.includes("healthy") ||
    lowerText.includes("salad") ||
    lowerText.includes("bowl") ||
    lowerText.includes("protein")
  ) {
    return "Healthy";
  }

  if (
    lowerText.includes("chinese") ||
    lowerText.includes("noodle") ||
    lowerText.includes("dumpling") ||
    lowerText.includes("hotpot")
  ) {
    return "Chinese";
  }

  if (
    lowerText.includes("korean") ||
    lowerText.includes("bbq") ||
    lowerText.includes("kimchi")
  ) {
    return "Korean";
  }

  if (
    lowerText.includes("pizza") ||
    lowerText.includes("pasta") ||
    lowerText.includes("italian")
  ) {
    return "Italian";
  }

  if (
    lowerText.includes("mexican") ||
    lowerText.includes("taco") ||
    lowerText.includes("burrito")
  ) {
    return "Mexican";
  }

  return "General";
}

function formatGooglePlaceType(type?: string): string {
  if (!type) {
    return "Restaurant";
  }

  return type
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function createGoogleMapsSearchUrl(name: string, address: string): string {
  const query = encodeURIComponent(`${name} ${address}`);

  return `https://www.google.com/maps/search/?api=1&query=${query}`;
}

function inferMealType(menu: MenuItem[]): RestaurantMealType {
  if (menu.length === 0) {
    return "normal";
  }

  const averageCalories =
    menu.reduce((sum, item) => sum + item.calories, 0) / menu.length;

  if (averageCalories <= 600) {
    return "healthy";
  }

  if (averageCalories >= 850) {
    return "cheat";
  }

  return "normal";
}

function inferProteinScore(menu: MenuItem[]): number {
  if (menu.length === 0) {
    return 60;
  }

  const averageProtein =
    menu.reduce((sum, item) => sum + item.protein, 0) / menu.length;

  return Math.round(Math.min(100, (averageProtein / 45) * 100));
}

function inferHealthScore(menu: MenuItem[]): number {
  if (menu.length === 0) {
    return 60;
  }

  const averageCalories =
    menu.reduce((sum, item) => sum + item.calories, 0) / menu.length;

  const averageFiber =
    menu.reduce((sum, item) => sum + item.fiber, 0) / menu.length;

  const averageFat =
    menu.reduce((sum, item) => sum + item.fat, 0) / menu.length;

  let score = 60;

  if (averageCalories <= 650) {
    score += 15;
  }

  if (averageFiber >= 6) {
    score += 15;
  }

  if (averageFat <= 25) {
    score += 10;
  }

  return Math.min(100, score);
}

function inferRestaurantTags(cuisine: string, category: string): string[] {
  const tags = new Set<string>();

  tags.add(cuisine);
  tags.add(category);

  if (cuisine === "Healthy") {
    tags.add("Healthy Meal");
  }

  if (cuisine === "Japanese") {
    tags.add("Balanced");
  }

  if (cuisine === "Indian") {
    tags.add("Curry");
  }

  if (cuisine === "Burger") {
    tags.add("Cheat Meal");
  }

  return Array.from(tags);
}