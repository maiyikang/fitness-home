export type RestaurantMealType = "healthy" | "normal" | "cheat";

export type MenuItem = {
  id: string;
  name: string;
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
  tags: string[];
};

export type Restaurant = {
  id: string;
  name: string;
  category: string;
  cuisine: string;
  mealType: RestaurantMealType;
  averageCalories: number;
  proteinScore: number;
  healthScore: number;
  matchScore: number;
  tags: string[];
  menu: MenuItem[];
};