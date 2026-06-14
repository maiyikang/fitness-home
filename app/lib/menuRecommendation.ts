import type { MenuItem, Restaurant } from "@/app/types/restaurant";
import type { UserProfile } from "@/app/types/profile";
import { calculateNutritionTargets } from "@/app/lib/nutrition";

export type MenuScoreBreakdown = {
  nutriQuality: number;
  goalMatch: number;
  caloriesMatch: number;
  macroMatch: number;
};

export type RecommendedMenuItem = MenuItem & {
  recommendationScore: number;
  recommendationReasons: string[];
  scoreBreakdown: MenuScoreBreakdown;
};

export function getRecommendedMenuItems(
  restaurant: Restaurant,
  profile: UserProfile
): RecommendedMenuItem[] {
  const nutritionTargets = calculateNutritionTargets(profile);
  const mealCalorieBudget = Math.round(nutritionTargets.calories / 3);

  return restaurant.menu
    .map((item) => {
      const scoreBreakdown = calculateMenuItemScoreBreakdown(
        item,
        profile,
        mealCalorieBudget
      );

      return {
        ...item,
        recommendationScore: calculateTotalMenuItemScore(scoreBreakdown),
        recommendationReasons: getMenuItemRecommendationReasons(
          item,
          profile,
          mealCalorieBudget
        ),
        scoreBreakdown,
      };
    })
    .sort((a, b) => b.recommendationScore - a.recommendationScore)
    .slice(0, 2);
}

function calculateMenuItemScoreBreakdown(
  item: MenuItem,
  profile: UserProfile,
  mealCalorieBudget: number
): MenuScoreBreakdown {
  const nutritionTargets = calculateNutritionTargets(profile);

  const proteinTargetPerMeal = Math.max(nutritionTargets.protein / 3, 1);
  const fiberTargetPerMeal = Math.max(nutritionTargets.fiber / 3, 1);
  const fatTargetPerMeal = Math.max(nutritionTargets.fat / 3, 1);

  const nutriQuality = calculateNutriQualityScore(
    item,
    proteinTargetPerMeal,
    fiberTargetPerMeal,
    fatTargetPerMeal
  );

  const goalMatch = calculateGoalMatchScore(
    item,
    profile,
    mealCalorieBudget,
    proteinTargetPerMeal,
    fiberTargetPerMeal
  );

  const caloriesMatch = calculateCaloriesMatchScore(
    item,
    mealCalorieBudget
  );

  const macroMatch = calculateMacroMatchScore(item, profile);

  return {
    nutriQuality,
    goalMatch,
    caloriesMatch,
    macroMatch,
  };
}

function calculateTotalMenuItemScore(
  breakdown: MenuScoreBreakdown
): number {
  return Math.round(
    breakdown.nutriQuality +
      breakdown.goalMatch +
      breakdown.caloriesMatch +
      breakdown.macroMatch
  );
}

function calculateNutriQualityScore(
  item: MenuItem,
  proteinTargetPerMeal: number,
  fiberTargetPerMeal: number,
  fatTargetPerMeal: number
): number {
  const proteinPart =
    Math.min(item.protein / proteinTargetPerMeal, 1) * 20;

  const fiberPart =
    Math.min(item.fiber / fiberTargetPerMeal, 1) * 10;

  const fatRatio = Math.max(
    0,
    1 - Math.abs(item.fat - fatTargetPerMeal) / fatTargetPerMeal
  );

  const fatPart = fatRatio * 10;

  return Math.round(
    Math.min(proteinPart + fiberPart + fatPart, 40)
  );
}

function calculateGoalMatchScore(
  item: MenuItem,
  profile: UserProfile,
  mealCalorieBudget: number,
  proteinTargetPerMeal: number,
  fiberTargetPerMeal: number
): number {
  let score = 0;

  if (profile.goal === "lose-fat") {
    const calorieScore =
      item.calories <= mealCalorieBudget
        ? 10
        : Math.max(
            0,
            10 *
              (1 -
                (item.calories - mealCalorieBudget) /
                  mealCalorieBudget)
          );

    const proteinScore =
      Math.min(item.protein / proteinTargetPerMeal, 1) * 10;

    const fiberScore =
      Math.min(item.fiber / fiberTargetPerMeal, 1) * 10;

    score = calorieScore + proteinScore + fiberScore;
  }

  if (profile.goal === "gain-muscle") {
    const proteinScore =
      Math.min(item.protein / proteinTargetPerMeal, 1) * 18;

    const calorieScore =
      item.calories >= mealCalorieBudget - 150
        ? 7
        : Math.max(
            0,
            7 *
              (item.calories /
                Math.max(mealCalorieBudget - 150, 1))
          );

    const carbScore =
      item.carbs >= 40 ? 5 : Math.min(item.carbs / 40, 1) * 5;

    score = proteinScore + calorieScore + carbScore;
  }

  if (profile.goal === "maintain") {
    const calorieDifference = Math.abs(
      item.calories - mealCalorieBudget
    );

    const calorieScore =
      Math.max(0, 1 - calorieDifference / mealCalorieBudget) * 15;

    const proteinScore =
      Math.min(item.protein / proteinTargetPerMeal, 1) * 8;

    const balanceScore = item.fat <= 30 && item.carbs >= 30 ? 7 : 4;

    score = calorieScore + proteinScore + balanceScore;
  }

  return Math.round(Math.min(score, 30));
}

function calculateCaloriesMatchScore(
  item: MenuItem,
  mealCalorieBudget: number
): number {
  const calorieDifference = Math.abs(
    item.calories - mealCalorieBudget
  );

  const score =
    20 * (1 - calorieDifference / Math.max(mealCalorieBudget, 1));

  return Math.round(Math.max(0, Math.min(score, 20)));
}

function calculateMacroMatchScore(
  item: MenuItem,
  profile: UserProfile
): number {
  const caloriesFromProtein = item.protein * 4;
  const caloriesFromCarbs = item.carbs * 4;
  const caloriesFromFat = item.fat * 9;

  const totalMacroCalories =
    caloriesFromProtein + caloriesFromCarbs + caloriesFromFat;

  if (totalMacroCalories <= 0) {
    return 0;
  }

  const proteinRatio = caloriesFromProtein / totalMacroCalories;
  const carbsRatio = caloriesFromCarbs / totalMacroCalories;
  const fatRatio = caloriesFromFat / totalMacroCalories;

  let targetProteinRatio = 0.3;
  let targetCarbsRatio = 0.4;
  let targetFatRatio = 0.3;

  if (profile.goal === "lose-fat") {
    targetProteinRatio = 0.35;
    targetCarbsRatio = 0.35;
    targetFatRatio = 0.3;
  }

  if (profile.goal === "gain-muscle") {
    targetProteinRatio = 0.35;
    targetCarbsRatio = 0.45;
    targetFatRatio = 0.2;
  }

  const proteinPenalty = Math.abs(proteinRatio - targetProteinRatio);
  const carbsPenalty = Math.abs(carbsRatio - targetCarbsRatio);
  const fatPenalty = Math.abs(fatRatio - targetFatRatio);

  const totalPenalty = proteinPenalty + carbsPenalty + fatPenalty;

  const score = 10 - totalPenalty * 10;

  return Math.round(Math.max(0, Math.min(score, 10)));
}

function getMenuItemRecommendationReasons(
  item: MenuItem,
  profile: UserProfile,
  mealCalorieBudget: number
): string[] {
  const reasons: string[] = [];
  const calorieDifference = Math.abs(
    item.calories - mealCalorieBudget
  );

  if (calorieDifference <= mealCalorieBudget * 0.15) {
    reasons.push("Close to your meal calorie budget.");
  } else if (item.calories < mealCalorieBudget) {
    reasons.push("Lower calorie option for this meal.");
  } else {
    reasons.push("Higher calorie option, better for flexible meals.");
  }

  if (profile.goal === "lose-fat") {
    if (item.calories <= mealCalorieBudget) {
      reasons.push("Good fit for fat loss.");
    }

    if (item.fiber >= 8) {
      reasons.push("Good fiber content for fullness.");
    }
  }

  if (profile.goal === "gain-muscle") {
    if (item.protein >= 40) {
      reasons.push("Strong protein choice for muscle gain.");
    } else if (item.protein >= 30) {
      reasons.push("Decent protein amount.");
    }
  }

  if (profile.goal === "maintain") {
    if (calorieDifference <= mealCalorieBudget * 0.2) {
      reasons.push("Balanced option for maintenance.");
    }
  }

  if (profile.mealType === "healthy") {
    if (item.tags.includes("Healthy")) {
      reasons.push("Matches your healthy meal preference.");
    }

    if (item.fat <= 25) {
      reasons.push("Moderate fat level.");
    }
  }

  if (profile.mealType === "cheat" && item.tags.includes("Cheat Meal")) {
    reasons.push("Matches your cheat meal preference.");
  }

  if (item.tags.includes("High Protein")) {
    reasons.push("Tagged as high protein.");
  }

  if (item.tags.includes("Low Calorie")) {
    reasons.push("Tagged as low calorie.");
  }

  if (reasons.length === 0) {
    reasons.push("Recommended as a balanced choice for this restaurant.");
  }

  return reasons.slice(0, 3);
}