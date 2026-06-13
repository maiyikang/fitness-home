export interface UserProfile {
  height: number;
  weight: number;
  age: number;

  gender: "male" | "female";

  activityLevel:
    | "low"
    | "medium"
    | "high";

  goal:
    | "lose-fat"
    | "maintain"
    | "gain-muscle";

  mealType:
    | "healthy"
    | "normal"
    | "cheat";

  favoriteCuisine: string[];
}