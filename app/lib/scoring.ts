import type { UserProfile } from "@/app/types/profile";
import type { Restaurant } from "@/app/types/restaurant";

import { isCuisineMatch } from "@/app/lib/cuisine";
import { getRecommendedMenuItems } from "@/app/lib/menuRecommendation";
import { calculateRestaurantHealthScore } from "@/app/lib/restaurantHealth";

export type MatchScoreReason = {
  label: string;
  points: number;
  type: "positive" | "negative" | "neutral";
};

export function calculateMatchScore(
  restaurant: Restaurant,
  profile: UserProfile
): number {
  const menuQualityScore = calculateMenuQualityScore(restaurant, profile);
  const restaurantHealthScore = calculateRestaurantHealthScore(
    restaurant
  ).score;
  const restaurantHealthWeightedScore =
    calculateRestaurantHealthWeightedScore(restaurantHealthScore);
  const cuisineMatchScore = calculateCuisineMatchScore(restaurant, profile);
  const mealTypeMatchScore = calculateMealTypeMatchScore(restaurant, profile);

  return Math.round(
    menuQualityScore +
      restaurantHealthWeightedScore +
      cuisineMatchScore +
      mealTypeMatchScore
  );
}

export function getMatchScoreReasons(
  restaurant: Restaurant,
  profile: UserProfile
): MatchScoreReason[] {
  const reasons: MatchScoreReason[] = [];

  const menuQualityScore = calculateMenuQualityScore(restaurant, profile);
  const restaurantHealthScore = calculateRestaurantHealthScore(
    restaurant
  ).score;
  const restaurantHealthWeightedScore =
    calculateRestaurantHealthWeightedScore(restaurantHealthScore);
  const cuisineMatchScore = calculateCuisineMatchScore(restaurant, profile);
  const mealTypeMatchScore = calculateMealTypeMatchScore(restaurant, profile);

  reasons.push({
    label: "Recommended menu quality",
    points: Math.round(menuQualityScore),
    type: "positive",
  });

  reasons.push({
    label: `Menu nutrition quality (${restaurantHealthScore}/100)`,
    points: Math.round(restaurantHealthWeightedScore),
    type: restaurantHealthScore >= 70 ? "positive" : "neutral",
  });

  if (cuisineMatchScore > 0) {
    reasons.push({
      label: "Matches your favorite cuisine",
      points: Math.round(cuisineMatchScore),
      type: "positive",
    });
  }

  if (mealTypeMatchScore > 0) {
    reasons.push({
      label: "Matches your selected meal type",
      points: Math.round(mealTypeMatchScore),
      type: "positive",
    });
  }

  if (restaurantHealthScore >= 85) {
    reasons.push({
      label: "Excellent overall menu health",
      points: 0,
      type: "positive",
    });
  } else if (restaurantHealthScore >= 70) {
    reasons.push({
      label: "Good overall menu health",
      points: 0,
      type: "positive",
    });
  } else if (restaurantHealthScore >= 55) {
    reasons.push({
      label: "Average overall menu health",
      points: 0,
      type: "neutral",
    });
  } else {
    reasons.push({
      label: "Menu nutrition quality could be improved",
      points: 0,
      type: "negative",
    });
  }

  return reasons;
}

function calculateMenuQualityScore(
  restaurant: Restaurant,
  profile: UserProfile
): number {
  const recommendedMenuItems = getRecommendedMenuItems(restaurant, profile);

  if (recommendedMenuItems.length === 0) {
    return 0;
  }

  const averageRecommendedMenuScore =
    recommendedMenuItems.reduce((sum, item) => {
      return sum + item.recommendationScore;
    }, 0) / recommendedMenuItems.length;

  return Math.min(50, averageRecommendedMenuScore * 0.5);
}

function calculateRestaurantHealthWeightedScore(
  restaurantHealthScore: number
): number {
  return Math.min(20, restaurantHealthScore * 0.2);
}

function calculateCuisineMatchScore(
  restaurant: Restaurant,
  profile: UserProfile
): number {
  const hasCuisineMatch = isCuisineMatch(
    profile.favoriteCuisine,
    restaurant.cuisine,
    restaurant.category,
    restaurant.tags
  );

  return hasCuisineMatch ? 15 : 0;
}

function calculateMealTypeMatchScore(
  restaurant: Restaurant,
  profile: UserProfile
): number {
  if (restaurant.mealType === profile.mealType) {
    return 15;
  }

  if (profile.mealType === "normal") {
    return 8;
  }

  return 0;
}