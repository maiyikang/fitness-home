import type { UserProfile } from "@/app/types/profile";

type RestaurantForScoring = {
  averageCalories: number;
  proteinScore: number;
  healthScore: number;
  category: string;
  tags: string[];
};

export type MatchScoreReason = {
  label: string;
  points: number;
  type: "positive" | "negative" | "neutral";
};

export function calculateMatchScore(
  restaurant: RestaurantForScoring,
  profile: UserProfile
): number {
  const reasons = getMatchScoreReasons(restaurant, profile);

  const totalScore = reasons.reduce((sum, reason) => {
    return sum + reason.points;
  }, 60);

  return Math.max(0, Math.min(100, totalScore));
}

export function getMatchScoreReasons(
  restaurant: RestaurantForScoring,
  profile: UserProfile
): MatchScoreReason[] {
  const reasons: MatchScoreReason[] = [];

  if (profile.goal === "lose-fat") {
    if (restaurant.averageCalories <= 650) {
      reasons.push({
        label: "Good for fat loss",
        points: 12,
        type: "positive",
      });
    } else {
      reasons.push({
        label: "Calories may be high for fat loss",
        points: -8,
        type: "negative",
      });
    }
  }

  if (profile.goal === "gain-muscle") {
    if (restaurant.proteinScore >= 8) {
      reasons.push({
        label: "High protein choice",
        points: 12,
        type: "positive",
      });
    } else {
      reasons.push({
        label: "Protein may be too low",
        points: -6,
        type: "negative",
      });
    }
  }

  if (profile.goal === "maintain") {
    if (
      restaurant.averageCalories >= 500 &&
      restaurant.averageCalories <= 850
    ) {
      reasons.push({
        label: "Balanced calories for maintenance",
        points: 8,
        type: "positive",
      });
    }
  }

  if (profile.mealType === "healthy") {
    if (restaurant.healthScore >= 8) {
      reasons.push({
        label: "Strong healthy meal match",
        points: 10,
        type: "positive",
      });
    } else {
      reasons.push({
        label: "Not the healthiest option",
        points: -5,
        type: "negative",
      });
    }
  }

  if (profile.mealType === "normal") {
    reasons.push({
      label: "Suitable for a normal meal",
      points: 5,
      type: "neutral",
    });
  }

  if (profile.mealType === "cheat") {
    reasons.push({
      label: "Flexible choice for a cheat meal",
      points: 5,
      type: "neutral",
    });
  }

  const lowerCategory = restaurant.category.toLowerCase();

  const hasCuisineMatch = profile.favoriteCuisine.some((cuisine) => {
    const lowerCuisine = cuisine.toLowerCase();

    return (
      lowerCuisine !== "normal" &&
      lowerCuisine !== "" &&
      lowerCategory.includes(lowerCuisine)
    );
  });

  if (hasCuisineMatch) {
    reasons.push({
      label: "Matches your favorite cuisine",
      points: 10,
      type: "positive",
    });
  }

  if (restaurant.tags.includes("High Protein")) {
    reasons.push({
      label: "Menu includes high protein options",
      points: 8,
      type: "positive",
    });
  }

  if (restaurant.tags.includes("Low Calorie")) {
    reasons.push({
      label: "Menu includes low calorie options",
      points: 8,
      type: "positive",
    });
  }

  if (restaurant.healthScore >= 8) {
    reasons.push({
      label: "Overall healthy restaurant",
      points: 7,
      type: "positive",
    });
  }

  return reasons;
}