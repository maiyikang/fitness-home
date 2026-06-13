import type { UserProfile } from "@/app/types/profile";

export type NutritionTargets = {
  calories: number;
  protein: number;
  carbs: number;
  fat: number;
  fiber: number;
};

export function calculateBMI(height: number, weight: number): number {
  const heightInMeters = height / 100;
  return Number((weight / (heightInMeters * heightInMeters)).toFixed(1));
}

export function calculateBMR(profile: UserProfile): number {
  if (profile.gender === "male") {
    return Math.round(
      10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
    );
  }

  return Math.round(
    10 * profile.weight + 6.25 * profile.height - 5 * profile.age - 161
  );
}

export function calculateTDEE(profile: UserProfile): number {
  const bmr = calculateBMR(profile);

  const activityMultiplier = {
    low: 1.2,
    medium: 1.55,
    high: 1.725,
  };

  return Math.round(bmr * activityMultiplier[profile.activityLevel]);
}

export function calculateTargetCalories(profile: UserProfile): number {
  const tdee = calculateTDEE(profile);

  if (profile.goal === "lose-fat") {
    return tdee - 400;
  }

  if (profile.goal === "gain-muscle") {
    return tdee + 300;
  }

  return tdee;
}

export function calculateProteinTarget(profile: UserProfile): number {
  if (profile.goal === "gain-muscle") {
    return Math.round(profile.weight * 2);
  }

  if (profile.goal === "lose-fat") {
    return Math.round(profile.weight * 1.8);
  }

  return Math.round(profile.weight * 1.6);
}

export function calculateFatTarget(profile: UserProfile): number {
  const targetCalories = calculateTargetCalories(profile);

  return Math.round((targetCalories * 0.25) / 9);
}

export function calculateCarbTarget(profile: UserProfile): number {
  const targetCalories = calculateTargetCalories(profile);
  const proteinTarget = calculateProteinTarget(profile);
  const fatTarget = calculateFatTarget(profile);

  const proteinCalories = proteinTarget * 4;
  const fatCalories = fatTarget * 9;

  return Math.max(
    0,
    Math.round((targetCalories - proteinCalories - fatCalories) / 4)
  );
}

export function calculateFiberTarget(profile: UserProfile): number {
  const targetCalories = calculateTargetCalories(profile);

  return Math.round((targetCalories / 1000) * 14);
}

export function calculateNutritionTargets(
  profile: UserProfile
): NutritionTargets {
  return {
    calories: calculateTargetCalories(profile),
    protein: calculateProteinTarget(profile),
    carbs: calculateCarbTarget(profile),
    fat: calculateFatTarget(profile),
    fiber: calculateFiberTarget(profile),
  };
}