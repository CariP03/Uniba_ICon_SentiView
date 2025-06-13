:- dynamic token_at/2.

% Base: if the word has a score, otherwise 0
token_base_score(Word, Score) :-
    word_score(Word, Score), !.
token_base_score(_, 0).

% Manage intensifiers and negators
token_effective_score(Index, ScoreEff) :-
    token_at(Index, Word),
    token_base_score(Word, Base),
    % Intensifier
    ( Index > 1,
      Prev is Index - 1,
      token_at(Prev, PrevW),
      intensifier(PrevW, M) ->
        B1 is Base * M ;
        B1 = Base ),
    % Negator
    ( Index > 1,
      Prev is Index - 1,
      token_at(Prev, PrevW2),
      negator(PrevW2) ->
        ScoreEff is -B1 ;
        ScoreEff = B1 ).

% Total Sum
sentiment_sum(Sum) :-
    findall(S, token_effective_score(_, S), Scores),
    sum_list(Scores, Sum).

% Non-zero tokens count 
sentiment_count_nonzero(Count) :-
    findall(1, (token_effective_score(_, S), S \= 0), L),
    length(L, Count).

% Total token count
sentiment_total_tokens(Tot) :-
    findall(1, token_at(_, _), L),
    length(L, Tot).