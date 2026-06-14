import type { UserProfile } from "@/app/types/profile";
import { calculateNutritionTargets } from "@/app/lib/nutrition";

type RestaurantForMealRecommendation = {
  averageCalories: number;
  proteinScore: number;
  healthScore: number;
  cuisine: string;
  mealType: string;
  tags: string[];
};

export type MealRecommendation = {
  title: string;
  calories: number;
  protein: number;
  message: string;
  reasons: string[];
};

export function getMealRecommendation(
  restaurant: RestaurantForMealRecommendation,
  profile: UserProfile
): MealRecommendation {
  const nutritionTargets = calculateNutritionTargets(profile);

  const suggestedMealCalories = Math.round(nutritionTargets.calories / 3);

  const calorieDifference = restaurant.averageCalories - suggestedMealCalories;

  const estimatedProtein = Math.round((restaurant.proteinScore / 100) * 55);

  const reasons: string[] = [];

  if (Math.abs(calorieDifference) <= 150) {
    reasons.push("Calories are close to your suggested meal budget.");
  }

  if (calorieDifference > 150) {
    reasons.push("This meal may be higher than your suggested meal budget.");
  }

  if (calorieDifference < -150) {
    reasons.push("This meal is lighter than your suggested meal budget.");
  }

  if (restaurant.proteinScore >= 80) {
    reasons.push("This restaurant is strong for protein.");
  }

  if (restaurant.healthScore >= 80) {
    reasons.push("This restaurant is a healthy choice.");
  }

  if (profile.mealType === restaurant.mealType) {
    reasons.push("This restaurant matches your selected meal type.");
  }

  if (
    profile.favoriteCuisine.some(
      (cuisine) =>
        cuisine.toLowerCase() === restaurant.cuisine.toLowerCase()
    )
  ) {
    reasons.push("This restaurant matches your favorite cuisine.");
  }

  if (reasons.length === 0) {
    reasons.push("This restaurant can still fit your plan in moderation.");
  }

  return {
    title: "Recommended Meal Strategy",
    calories: restaurant.averageCalories,
    protein: estimatedProtein,
    message: `Your suggested meal budget is about ${suggestedMealCalories} kcal.`,
    reasons,
  };
}